"""
简化版推理脚本 - 快速对单个患者进行预测 (2.5D版本)
"""
import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from model_v2 import ResNet25DMultiTask, PredictionsAggregator


def load_model(checkpoint_path, device='cpu'):
    """加载2.5D模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get('args', {})

    model = ResNet25DMultiTask(
        num_seg_classes=1,
        num_cls_classes=2,
        slices_per_input=args_dict.get('slices_per_input', 3),
        use_attention=args_dict.get('use_attention', True),
        pretrained=False,
        dropout=args_dict.get('dropout', 0.3),
        use_roi_masking=args_dict.get('use_roi_masking', True)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, args_dict


def preprocess_slice(img_slice, target_size=256):
    """预处理单个切片"""
    # 归一化
    mean_val = np.mean(img_slice)
    std_val = np.std(img_slice) + 1e-8
    img_slice = (img_slice - mean_val) / std_val

    # 转换为张量并resize
    img_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).float()
    img_tensor = F.interpolate(img_tensor, size=(target_size, target_size),
                                mode='bilinear', align_corners=False)

    return img_tensor.squeeze(0)  # (1, H, W)


def predict(model, image_path, organ_mask_path, device='cpu', slices_per_input=3):
    """
    对nii图像进行2.5D预测

    Args:
        model: 2.5D模型
        image_path: 图像路径
        organ_mask_path: 器官mask路径（用于ROI Masking）
        device: 设备
        slices_per_input: 2.5D输入切片数

    Returns:
        dict: 包含分类预测结果和分割概率图
    """
    # 加载图像和mask
    sitk_image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(sitk_image)
    sitk_mask = sitk.ReadImage(organ_mask_path)
    organ_mask = sitk.GetArrayFromImage(sitk_mask)

    if len(image.shape) == 4:
        image = image[:, :, :, 0]

    print(f"Image shape: {image.shape}")

    # 多器官ROI提取（与评估脚本一致）
    multi_organ = (organ_mask == 1) | (organ_mask == 2) | (organ_mask == 3)
    if not np.any(multi_organ):
        d, h, w = image.shape
        bbox = (d//4, 3*d//4, h//4, 3*h//4, w//4, 3*w//4)
    else:
        z_idx, y_idx, x_idx = np.where(multi_organ)
        bbox = (z_idx.min(), z_idx.max(), y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max())

    z_min, z_max = bbox[0], bbox[1]
    z_extension = 3
    z_start = max(0, z_min - z_extension)
    z_end = min(image.shape[0], z_max + z_extension + 1)

    image_crop = image[z_start:z_end, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1]
    pancreas_mask_crop = (organ_mask[z_start:z_end, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1] == 1).astype(np.float32)

    # 归一化 (窗宽窗位)
    ww, wl = 400, 40
    min_val, max_val = wl - ww//2, wl + ww//2
    image_crop = np.clip(image_crop, min_val, max_val)
    image_crop = (image_crop - min_val) / (max_val - min_val)

    # 重采样到固定尺寸 (32, 256, 256)
    target_d, target_h, target_w = 32, 256, 256
    image_tensor = torch.from_numpy(image_crop).float().unsqueeze(0).unsqueeze(0)
    pancreas_mask_tensor = torch.from_numpy(pancreas_mask_crop).float().unsqueeze(0).unsqueeze(0)

    image_resampled = F.interpolate(image_tensor, size=(target_d, target_h, target_w),
                                     mode='trilinear', align_corners=False)
    pancreas_mask_resampled = F.interpolate(pancreas_mask_tensor, size=(target_d, target_h, target_w),
                                             mode='nearest')

    # 创建聚合器
    aggregator = PredictionsAggregator(
        depth=target_d,
        slices_per_input=slices_per_input,
        stride=1
    )

    # 2.5D推理 - 收集所有slice group的预测
    all_cls_probs = []
    seg_outputs = []
    slice_indices_list = []

    with torch.no_grad():
        for d in range(0, target_d - slices_per_input + 1):
            indices = list(range(d, d + slices_per_input))
            # 2.5D输入: (1, slices_per_input, H, W)
            input_25d = image_resampled[0, 0, d:d+slices_per_input, :, :].unsqueeze(0).to(device)
            # 取中间slice的mask用于ROI Masking
            mid_idx = d + slices_per_input // 2
            mask_2d = pancreas_mask_resampled[0, 0, mid_idx, :, :].unsqueeze(0).unsqueeze(0).to(device)

            # 传入mask进行ROI Masking
            outputs = model(input_25d, mask_2d)

            cls_prob = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
            seg_out = torch.sigmoid(outputs['segmentation']).cpu().numpy()[0, 0]

            all_cls_probs.append(cls_prob)
            seg_outputs.append(seg_out)
            slice_indices_list.append(indices)

    # 使用聚合器融合结果
    aggregated_cls = aggregator.aggregate_classification(all_cls_probs)
    aggregated_seg = aggregator.aggregate_segmentation(
        seg_outputs, slice_indices_list, (target_d, target_h, target_w)
    )

    # 使用与训练一致的阈值0.4
    pred_prob = float(aggregated_cls[1])
    cls_pred = 1 if pred_prob >= 0.4 else 0
    cls_confidence = float(aggregated_cls.max())

    return {
        'gata6_prediction': cls_pred,
        'gata6_confidence': cls_confidence,
        'gata6_probability_negative': float(aggregated_cls[0]),
        'gata6_probability_positive': pred_prob,
        'segmentation_volume': aggregated_seg,
        'num_predictions': len(all_cls_probs),
        'bbox': bbox
    }


def main():
    parser = argparse.ArgumentParser(description='Predict GATA6 and pancreas segmentation (2.5D)')
    parser.add_argument('--image', type=str, required=True, help='Path to image.nii file')
    parser.add_argument('--organ_mask', type=str, required=True, help='Path to organ_label.nii file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_seg', type=str, help='Output path for segmentation (optional)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--slices_per_input', type=int, default=3, help='Number of slices per 2.5D input')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # 使用模型配置或命令行参数
    slices_per_input = model_args.get('slices_per_input', args.slices_per_input)

    # 预测
    print(f"\nProcessing: {args.image}")
    results = predict(model, args.image, args.organ_mask, device, slices_per_input)

    # 输出结果
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"GATA6 Mutation Prediction: {'POSITIVE (1)' if results['gata6_prediction'] == 1 else 'NEGATIVE (0)'}")
    print(f"  Confidence: {results['gata6_confidence']:.4f}")
    print(f"  Probability (Negative): {results['gata6_probability_negative']:.4f}")
    print(f"  Probability (Positive): {results['gata6_probability_positive']:.4f}")
    print(f"  (Using threshold: 0.4)")
    print(f"\nSegmentation: Generated from {results['num_predictions']} overlapping predictions")
    print("="*50)

    # 保存分割结果
    if args.output_seg:
        reference = sitk.ReadImage(args.image)
        seg_sitk = sitk.GetImageFromArray(results['segmentation_volume'].astype(np.float32))
        seg_sitk.SetSpacing(reference.GetSpacing())
        seg_sitk.SetOrigin(reference.GetOrigin())
        seg_sitk.SetDirection(reference.GetDirection())
        sitk.WriteImage(seg_sitk, args.output_seg)
        print(f"\nSegmentation saved to: {args.output_seg}")


if __name__ == '__main__':
    main()
