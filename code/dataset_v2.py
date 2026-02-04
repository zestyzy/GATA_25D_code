"""
Case级别的2.5D数据集
- 多器官（胰腺+肝脏+脾脏）确定ROI
- 固定层数：ROI上下各扩展3层
- 重采样到固定尺寸
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PancreasCaseDataset(Dataset):
    """
    Case级别的2.5D胰腺分割和GATA6分类数据集

    处理流程:
    1. 加载多器官mask (胰腺=1, 肝脏=2, 脾脏=3)
    2. 计算三个器官的联合bounding box
    3. 在Z方向扩展3层
    4. 裁剪ROI并重采样到固定尺寸
    """

    # 器官label定义
    ORGAN_LABELS = {
        'pancreas': 1,
        'liver': 2,
        'spleen': 3
    }

    def __init__(self, data_root, label_file, target_size=(256, 256, 32),
                 z_extension=3, transform=None, normalize_mode='window'):
        """
        Args:
            data_root: 数据根目录
            label_file: GATA6标签文件
            target_size: 目标尺寸 (H, W, D)，所有case将被重采样到此尺寸
            z_extension: Z方向扩展层数
            transform: 数据增强
            normalize_mode: 归一化方式 ('window'或'organ')
        """
        self.data_root = data_root
        self.target_size = target_size  # (H, W, D)
        self.z_extension = z_extension
        self.transform = transform
        self.normalize_mode = normalize_mode

        # 读取GATA6标签
        self.labels_df = pd.read_excel(label_file)
        self.labels_dict = dict(zip(
            self.labels_df['case_id'].values,
            self.labels_df['GATA6'].values
        ))

        # 构建样本列表
        self.samples = self._build_samples()
        print(f"Dataset loaded: {len(self.samples)} valid cases")
        print(f"Target size: {target_size}")

    def _build_samples(self):
        """构建有效样本列表"""
        samples = []
        patient_dirs = sorted([d for d in os.listdir(self.data_root)
                              if os.path.isdir(os.path.join(self.data_root, d))])

        for patient_dir in patient_dirs:
            try:
                case_id = int(patient_dir[:2])
            except ValueError:
                continue

            if case_id not in self.labels_dict:
                continue

            patient_path = os.path.join(self.data_root, patient_dir)
            image_path = os.path.join(patient_path, 'image.nii')
            organ_mask_path = os.path.join(patient_path, 'organ_label.nii')
            lesion_mask_path = os.path.join(patient_path, 'label.nii')

            if not all(os.path.exists(p) for p in [image_path, organ_mask_path, lesion_mask_path]):
                continue

            gata6_label = self.labels_dict[case_id]

            samples.append({
                'case_id': case_id,
                'patient_dir': patient_dir,
                'image_path': image_path,
                'organ_mask_path': organ_mask_path,
                'lesion_mask_path': lesion_mask_path,
                'gata6_label': gata6_label
            })

        return samples

    def _load_nii(self, path):
        """加载nii文件"""
        sitk_image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(sitk_image)  # (D, H, W)
        spacing = sitk_image.GetSpacing()  # (x, y, z)
        origin = sitk_image.GetOrigin()
        return array, spacing, origin

    def _get_multi_organ_bbox(self, organ_mask, organ_labels=[1, 2, 3]):
        """
        获取多器官的联合bounding box
        Args:
            organ_mask: 器官mask (D, H, W)
            organ_labels: 要包含的器官label列表 [1,2,3]
        Returns:
            (z_min, z_max, y_min, y_max, x_min, x_max)
        """
        # 创建多器官mask
        multi_organ = np.zeros_like(organ_mask, dtype=bool)
        for label in organ_labels:
            multi_organ |= (organ_mask == label)

        if not np.any(multi_organ):
            # 如果没有找到器官，返回整个图像的中心区域
            d, h, w = organ_mask.shape
            return (d//4, 3*d//4, h//4, 3*h//4, w//4, 3*w//4)

        # 找到非零位置的索引
        z_indices, y_indices, x_indices = np.where(multi_organ)

        z_min, z_max = z_indices.min(), z_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        return (z_min, z_max, y_min, y_max, x_min, x_max)

    def _crop_and_extend(self, array, bbox, z_extension):
        """
        根据bbox裁剪并在Z方向扩展
        Args:
            array: 输入数组 (D, H, W)
            bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
            z_extension: Z方向扩展层数
        Returns:
            裁剪后的数组
        """
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        d, h, w = array.shape

        # Z方向扩展
        z_start = max(0, z_min - z_extension)
        z_end = min(d, z_max + z_extension + 1)

        # 裁剪
        cropped = array[z_start:z_end, y_min:y_max+1, x_min:x_max+1]

        return cropped, (z_start, z_end)

    def _resample_3d(self, array, target_size, is_mask=False):
        """
        3D重采样到目标尺寸
        Args:
            array: 输入数组 (D, H, W)
            target_size: (H, W, D)
            is_mask: 是否是mask（使用最近邻插值）
        Returns:
            重采样后的数组
        """
        import torch.nn.functional as F

        # 添加batch和channel: (1, 1, D, H, W)
        tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)

        # 目标尺寸: (D, H, W) -> 需要变成 (D_out, H_out, W_out)
        target_d, target_h, target_w = target_size[2], target_size[0], target_size[1]

        # 使用trilinear或nearest插值
        mode = 'nearest' if is_mask else 'trilinear'
        align_corners = None if is_mask else False

        if mode == 'trilinear':
            resized = F.interpolate(tensor, size=(target_d, target_h, target_w),
                                   mode=mode, align_corners=align_corners)
        else:
            resized = F.interpolate(tensor, size=(target_d, target_h, target_w), mode=mode)

        return resized.squeeze(0).squeeze(0).numpy()

    def _normalize_intensity(self, image, organ_mask=None):
        """
        强度归一化
        Args:
            image: CT图像
            organ_mask: 胰腺mask (用于基于器官的归一化)
        """
        if self.normalize_mode == 'window':
            # CT窗宽窗位归一化 (腹部常用: WW=400, WL=40)
            ww, wl = 400, 40
            min_val = wl - ww // 2
            max_val = wl + ww // 2
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val)

        elif self.normalize_mode == 'organ' and organ_mask is not None:
            # 基于胰腺区域的归一化
            if np.any(organ_mask > 0):
                organ_pixels = image[organ_mask > 0]
                mean = np.mean(organ_pixels)
                std = np.std(organ_pixels) + 1e-8
                image = (image - mean) / std
            else:
                mean = np.mean(image)
                std = np.std(image) + 1e-8
                image = (image - mean) / std
        else:
            # 全局z-score
            mean = np.mean(image)
            std = np.std(image) + 1e-8
            image = (image - mean) / std

        return image

    def _apply_spatial_transform_sync(self, image, pancreas_mask, lesion_mask):
        """
        同步应用空间变换（2.5D专用修复）
        关键修复：image使用双线性插值，mask使用最近邻插值保持二值性

        Args:
            image: (H, W) tensor
            pancreas_mask: (H, W) tensor
            lesion_mask: (H, W) tensor
        Returns:
            变换后的image, pancreas_mask, lesion_mask
        """
        from torchvision import transforms
        from torchvision.transforms import functional as F

        # 确保是tensor并添加channel维度用于transform
        img = image.unsqueeze(0) if image.dim() == 2 else image
        pan_mask = pancreas_mask.unsqueeze(0) if pancreas_mask.dim() == 2 else pancreas_mask
        les_mask = lesion_mask.unsqueeze(0) if lesion_mask.dim() == 2 else lesion_mask

        h, w = img.shape[-2:]

        # 1. 随机水平翻转 (50%概率)
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            pan_mask = F.hflip(pan_mask)
            les_mask = F.hflip(les_mask)

        # 2. 随机垂直翻转 (50%概率)
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            pan_mask = F.vflip(pan_mask)
            les_mask = F.vflip(les_mask)

        # 3. 随机旋转 (-15到15度)
        angle = torch.empty(1).uniform_(-15, 15).item()
        # image使用双线性插值
        img = F.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        # mask使用最近邻插值保持二值
        pan_mask = F.rotate(pan_mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        les_mask = F.rotate(les_mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # 4. 随机仿射变换 (平移和缩放)
        tx = torch.empty(1).uniform_(-0.1, 0.1).item() * w
        ty = torch.empty(1).uniform_(-0.1, 0.1).item() * h
        scale = torch.empty(1).uniform_(0.9, 1.1).item()

        img = F.affine(img, angle=0, translate=(tx, ty), scale=scale, shear=0,
                       interpolation=transforms.InterpolationMode.BILINEAR)
        pan_mask = F.affine(pan_mask, angle=0, translate=(tx, ty), scale=scale, shear=0,
                            interpolation=transforms.InterpolationMode.NEAREST)
        les_mask = F.affine(les_mask, angle=0, translate=(tx, ty), scale=scale, shear=0,
                            interpolation=transforms.InterpolationMode.NEAREST)

        # 二值化mask（确保没有插值产生的中间值）
        pan_mask = (pan_mask > 0.5).float()
        les_mask = (les_mask > 0.5).float()

        return img.squeeze(0), pan_mask.squeeze(0), les_mask.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个case的数据
        Returns:
            dict: 包含以下键
                - image: (1, D, H, W) 重采样后的图像
                - pancreas_mask: (1, D, H, W) 胰腺分割mask
                - lesion_mask: (1, D, H, W) 病灶mask
                - gata6_label: GATA6分类标签
                - case_id: 患者ID
                - bbox_info: 裁剪信息
        """
        sample_info = self.samples[idx]

        # 加载数据
        image, img_spacing, img_origin = self._load_nii(sample_info['image_path'])
        organ_mask, _, _ = self._load_nii(sample_info['organ_mask_path'])
        lesion_mask, _, _ = self._load_nii(sample_info['lesion_mask_path'])

        # 获取多器官ROI
        bbox = self._get_multi_organ_bbox(organ_mask, organ_labels=[1, 2, 3])
        z_min, z_max = bbox[0], bbox[1]

        # 裁剪并扩展
        image_crop, (z_start, z_end) = self._crop_and_extend(image, bbox, self.z_extension)
        organ_crop, _ = self._crop_and_extend(organ_mask, bbox, self.z_extension)
        lesion_crop, _ = self._crop_and_extend(lesion_mask, bbox, self.z_extension)

        # 提取胰腺mask (label=1)
        pancreas_crop = (organ_crop == self.ORGAN_LABELS['pancreas']).astype(np.float32)

        # 归一化图像
        image_crop = self._normalize_intensity(image_crop, pancreas_crop)

        # 重采样到固定尺寸
        image_resampled = self._resample_3d(image_crop, self.target_size, is_mask=False)
        pancreas_resampled = self._resample_3d(pancreas_crop, self.target_size, is_mask=True)
        lesion_resampled = self._resample_3d(lesion_crop, self.target_size, is_mask=True)

        # 转换为torch张量: (D, H, W) -> (1, D, H, W)
        image_tensor = torch.from_numpy(image_resampled).float().unsqueeze(0)
        pancreas_tensor = torch.from_numpy(pancreas_resampled).float().unsqueeze(0)
        lesion_tensor = torch.from_numpy(lesion_resampled).float().unsqueeze(0)

        # 数据增强 (空间域) - 2.5D专用修复版
        # 关键修复：mask使用最近邻插值保持二值性，image使用双线性插值
        if self.transform is not None:
            image_aug = []
            pancreas_aug = []
            lesion_aug = []

            for d in range(image_tensor.shape[1]):  # 遍历D维度
                img_d = image_tensor[0, d, :, :]  # (H, W)
                pan_d = pancreas_tensor[0, d, :, :]  # (H, W)
                les_d = lesion_tensor[0, d, :, :]  # (H, W)

                # 使用albumentations风格的同步变换（如果可用）
                # 或者使用torchvision.functional进行手动同步变换
                img_aug, pan_aug, les_aug = self._apply_spatial_transform_sync(
                    img_d, pan_d, les_d
                )

                image_aug.append(img_aug.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)
                pancreas_aug.append(pan_aug.unsqueeze(0).unsqueeze(0))
                lesion_aug.append(les_aug.unsqueeze(0).unsqueeze(0))

            image_tensor = torch.cat(image_aug, dim=1)  # (1, D, H, W)
            pancreas_tensor = torch.cat(pancreas_aug, dim=1)
            lesion_tensor = torch.cat(lesion_aug, dim=1)

        return {
            'image': image_tensor,  # (1, D, H, W)
            'pancreas_mask': pancreas_tensor,  # (1, D, H, W)
            'lesion_mask': lesion_tensor,  # (1, D, H, W)
            'gata6_label': torch.tensor(sample_info['gata6_label'], dtype=torch.long),
            'case_id': sample_info['case_id'],
            'bbox_info': {
                'original_bbox': bbox,
                'crop_range': (z_start, z_end),
                'original_shape': image.shape,
                'resampled_shape': self.target_size
            }
        }


class ResNet25DInputTransform:
    """
    将3D volume转换为2.5D输入格式用于ResNet18
    将D维度转换为通道维度
    """

    @staticmethod
    def volume_to_25d(volume_3d, slice_groups=None):
        """
        将3D volume转换为2.5D格式
        Args:
            volume_3d: (1, D, H, W) 或 (D, H, W)
            slice_groups: 如何将slice分组，None表示所有slice作为独立通道
                         例如: [(0,1,2), (3,4,5), ...] 表示每3个slice一组
        Returns:
            tensor: (C, H, W) 其中C=D或C=len(slice_groups)
        """
        if volume_3d.dim() == 4:
            volume_3d = volume_3d.squeeze(0)  # (D, H, W)

        d, h, w = volume_3d.shape

        if slice_groups is None:
            # 每个slice作为一个通道
            return volume_3d  # (D, H, W)
        else:
            # 按组聚合slice
            grouped = []
            for group in slice_groups:
                # 取组内slice的平均
                group_slice = volume_3d[group, :, :].mean(dim=0, keepdim=True)
                grouped.append(group_slice)
            return torch.cat(grouped, dim=0)  # (C, H, W)


class Slice2p5DDataset(Dataset):
    """
    2.5D数据集: 将3D case转换为多个2.5D slice group
    适用于需要大batch size的训练
    """

    def __init__(self, base_dataset, slices_per_input=3, stride=1):
        """
        Args:
            base_dataset: PancreasCaseDataset实例
            slices_per_input: 每个2.5D输入包含的连续slice数
            stride: 滑动窗口步长
        """
        self.base_dataset = base_dataset
        self.slices_per_input = slices_per_input
        self.stride = stride

        # 预计算所有可能的slice group
        self.samples = []
        target_d = base_dataset.target_size[2]  # D维度

        for case_idx in range(len(base_dataset)):
            # 生成slice group索引
            for start_idx in range(0, target_d - slices_per_input + 1, stride):
                slice_indices = list(range(start_idx, start_idx + slices_per_input))
                self.samples.append({
                    'case_idx': case_idx,
                    'slice_indices': slice_indices
                })

        print(f"2.5D Dataset: {len(base_dataset)} cases -> {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        case_idx = sample_info['case_idx']
        slice_indices = sample_info['slice_indices']

        # 获取完整case
        case_data = self.base_dataset[case_idx]

        # 提取指定slice
        image_3d = case_data['image']  # (1, D, H, W)
        pancreas_mask_3d = case_data['pancreas_mask']
        lesion_mask_3d = case_data['lesion_mask']

        # 构建2.5D输入: (slices_per_input, H, W)
        image_25d = image_3d[0, slice_indices, :, :]  # (slices_per_input, H, W)

        # 对应位置的分割标签 (取中间slice的mask)
        mid_idx = len(slice_indices) // 2
        pancreas_mask = pancreas_mask_3d[0, slice_indices[mid_idx], :, :]  # (H, W)
        lesion_mask = lesion_mask_3d[0, slice_indices[mid_idx], :, :]

        # 提取当前slice group对应的3D mask区域用于Dice计算
        # 使用slice_indices范围内的完整mask
        z_start = max(0, slice_indices[0] - 1)
        z_end = min(pancreas_mask_3d.shape[1], slice_indices[-1] + 2)
        pancreas_mask_3d_region = pancreas_mask_3d[0, z_start:z_end, :, :]  # (local_D, H, W)

        return {
            'image': image_25d,  # (slices_per_input, H, W)
            'pancreas_mask': pancreas_mask,  # (H, W) - 中间slice的mask用于训练
            'pancreas_mask_3d': pancreas_mask_3d[0],  # (D, H, W) - 完整3D mask用于验证Dice
            'pancreas_mask_3d_region': pancreas_mask_3d_region,  # (local_D, H, W) - 当前区域mask
            'slice_indices': torch.tensor(slice_indices),
            'z_offset': z_start,  # 记录z方向偏移用于聚合
            'lesion_mask': lesion_mask,  # (H, W)
            'gata6_label': case_data['gata6_label'],
            'case_id': case_data['case_id']
        }


class CaseLevel2p5DDataset(Dataset):
    """
    Case级别的2.5D数据集 - 简化版本
    每个case生成不重叠的2.5D输入组

    例如: depth=32, slices_per_input=4 -> 8个样本 (0-3, 4-7, ..., 28-31)
    """

    def __init__(self, base_dataset, slices_per_input=4):
        """
        Args:
            base_dataset: PancreasCaseDataset实例
            slices_per_input: 每个2.5D输入包含的连续slice数
                              需要能被base_dataset.target_size[2]整除
        """
        self.base_dataset = base_dataset
        self.slices_per_input = slices_per_input

        depth = base_dataset.target_size[2]

        if depth % slices_per_input != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by slices_per_input ({slices_per_input}). "
                f"Suggested values: {self._suggest_divisors(depth)}"
            )

        self.groups_per_case = depth // slices_per_input

        # 预计算样本
        self.samples = []
        for case_idx in range(len(base_dataset)):
            for g in range(self.groups_per_case):
                start_idx = g * slices_per_input
                slice_indices = list(range(start_idx, start_idx + slices_per_input))
                self.samples.append({
                    'case_idx': case_idx,
                    'slice_indices': slice_indices,
                    'group_id': g
                })

        print(f"CaseLevel2.5D Dataset: {len(base_dataset)} cases x {self.groups_per_case} groups = {len(self.samples)} samples")

    def _suggest_divisors(self, n):
        """找到n的所有因数"""
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        case_idx = sample_info['case_idx']
        slice_indices = sample_info['slice_indices']

        case_data = self.base_dataset[case_idx]

        image_3d = case_data['image']  # (1, D, H, W)
        pancreas_mask_3d = case_data['pancreas_mask']
        lesion_mask_3d = case_data['lesion_mask']

        # 提取2.5D输入
        image_25d = image_3d[0, slice_indices, :, :]  # (slices_per_input, H, W)

        # 分割标签: 使用中间slice的mask
        mid_idx = len(slice_indices) // 2
        pancreas_mask = pancreas_mask_3d[0, slice_indices[mid_idx], :, :]  # (H, W)
        lesion_mask = lesion_mask_3d[0, slice_indices[mid_idx], :, :]

        return {
            'image': image_25d,
            'pancreas_mask': pancreas_mask,
            'lesion_mask': lesion_mask,
            'gata6_label': case_data['gata6_label'],
            'case_id': case_data['case_id'],
            'group_id': sample_info['group_id'],
            'slice_indices': torch.tensor(slice_indices)
        }


def get_transforms(phase='train'):
    """获取数据增强变换"""
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    else:
        return None


if __name__ == '__main__':
    # 测试数据集
    data_root = '/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/zs'
    label_file = '/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/GATA6_label.xlsx'

    # 创建case级别数据集
    dataset = PancreasCaseDataset(
        data_root=data_root,
        label_file=label_file,
        target_size=(256, 256, 32),  # (H, W, D)
        z_extension=3,
        normalize_mode='window'
    )

    print(f"\nTotal cases: {len(dataset)}")

    # 测试获取一个case
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")  # (1, D, H, W)
    print(f"Pancreas mask shape: {sample['pancreas_mask'].shape}")
    print(f"GATA6 label: {sample['gata6_label']}")
    print(f"Case ID: {sample['case_id']}")
    print(f"Bbox info: {sample['bbox_info']}")

    # 创建2.5D数据集
    dataset_25d = Slice2p5DDataset(dataset, slices_per_input=3, stride=1)
    print(f"\n2.5D samples: {len(dataset_25d)}")

    # 测试获取2.5D样本
    sample_25d = dataset_25d[0]
    print(f"\n2.5D sample:")
    print(f"  Image shape: {sample_25d['image'].shape}")  # (3, H, W)
    print(f"  Mask shape: {sample_25d['pancreas_mask'].shape}")  # (H, W)
    print(f"  Case ID: {sample_25d['case_id']}")
