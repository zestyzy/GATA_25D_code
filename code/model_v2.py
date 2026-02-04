"""
基于ResNet18的2.5D多任务模型
输入: 多slice作为通道 (如3个连续slice作为3通道)
任务: 胰腺分割 + GATA6基因二分类
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class DecoderBlock(nn.Module):
    """上采样解码块"""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                     stride=2, padding=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SegmentationHead(nn.Module):
    """分割头"""
    def __init__(self, in_channels, num_classes=1, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        return x


class AttentionModule(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet25DMultiTask(nn.Module):
    """
    2.5D多任务ResNet18模型

    输入: (B, C_2.5d, H, W) 其中C_2.5d是连续slice数(如3)
    输出:
        - segmentation: (B, 1, H, W) 胰腺分割
        - classification: (B, num_cls_classes) GATA6分类

    预训练权重支持:
        - pretrained=True: 使用ImageNet预训练权重
        - pretrained=False: 不使用预训练权重
        - pretrained="path/to/weights.pth": 加载自定义权重文件
    """

    def __init__(self, num_seg_classes=1, num_cls_classes=2,
                 slices_per_input=3, use_attention=True,
                 pretrained=True, dropout=0.3, use_roi_masking=True):
        super().__init__()

        self.slices_per_input = slices_per_input
        self.use_roi_masking = use_roi_masking

        # 加载预训练ResNet18
        if isinstance(pretrained, str):
            # 加载自定义权重文件
            backbone = resnet18(weights=None)
            self._load_custom_weights(backbone, pretrained)
            use_pretrained_conv = True
        elif pretrained:
            # 使用ImageNet预训练权重
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
            use_pretrained_conv = True
        else:
            # 不使用预训练权重
            backbone = resnet18(weights=None)
            use_pretrained_conv = False

        # 修改第一层以适应2.5D输入
        # 将3通道卷积改为slices_per_input通道
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(slices_per_input, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        if use_pretrained_conv:
            # 策略: 将ImageNet权重复制到每个输入通道
            with torch.no_grad():
                # 原始权重: (64, 3, 7, 7)
                # 新权重: (64, slices_per_input, 7, 7)
                # 将3通道权重平均，然后复制到所有输入通道
                mean_weight = original_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                # 重复slices_per_input次
                new_weight = mean_weight.repeat(1, slices_per_input, 1, 1)
                # 归一化，保持数值范围一致
                new_weight = new_weight / slices_per_input * 3
                self.conv1.weight = nn.Parameter(new_weight)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # 编码器层
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        # ========== 分割分支 ==========
        self.seg_bridge = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec4 = DecoderBlock(512, 256, skip_channels=256)
        self.dec3 = DecoderBlock(256, 128, skip_channels=128)
        self.dec2 = DecoderBlock(128, 64, skip_channels=64)
        self.dec1 = DecoderBlock(64, 64, skip_channels=64)

        self.seg_final_up = nn.ConvTranspose2d(64, 64, kernel_size=4,
                                                stride=2, padding=1)
        self.seg_head = SegmentationHead(64, num_seg_classes, dropout=0.1)

        # ========== 分类分支 ==========
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(512)

        self.cls_gap = nn.AdaptiveAvgPool2d(1)

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_cls_classes)
        )

        # 分割辅助分类
        self.seg_aux_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(32, num_cls_classes)
        )

        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]))

        self._initialize_weights()

    def _load_custom_weights(self, model, weights_path):
        """加载自定义预训练权重"""
        print(f"Loading custom pretrained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

        # 处理不同格式的权重文件
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 移除可能的'module.'前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Custom weights loaded successfully")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def masked_global_average_pool(self, features, mask):
        """
        使用mask进行全局平均池化，只在mask区域内池化

        Args:
            features: (B, C, H, W)
            mask: (B, 1, H, W) - 胰腺mask (0或1)
        Returns:
            pooled: (B, C)
        """
        # 确保mask尺寸与features匹配
        if features.shape[2:] != mask.shape[2:]:
            mask = F.interpolate(mask, size=features.shape[2:], mode='nearest')

        # mask: (B, 1, H, W) -> (B, 1, H, W)
        # features: (B, C, H, W)

        # 计算mask区域内的总和
        masked_features = features * mask  # (B, C, H, W)
        sum_features = masked_features.sum(dim=(2, 3))  # (B, C)

        # 计算mask区域内的像素数
        mask_sum = mask.sum(dim=(2, 3))  # (B, 1)
        mask_sum = mask_sum.clamp(min=1)  # 避免除零

        # 平均
        pooled = sum_features / mask_sum  # (B, C)

        return pooled

    def forward(self, x, pancreas_mask=None):
        """
        Args:
            x: (B, slices_per_input, H, W)
            pancreas_mask: (B, 1, H, W) - 可选，用于ROI Masking分类
        Returns:
            dict: 包含分割和分类输出
        """
        input_size = x.shape[2:]

        # ========== 编码器 ==========
        x0 = self.conv1(x)      # (B, 64, H/2, W/2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0_pool = self.maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.layer1(x0_pool)   # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)        # (B, 128, H/8, W/8)
        x3 = self.layer3(x2)        # (B, 256, H/16, W/16)
        x4 = self.layer4(x3)        # (B, 512, H/32, W/32)

        # ========== 分割分支 ==========
        seg = self.seg_bridge(x4)
        seg = self.dec4(seg, x3)
        seg = self.dec3(seg, x2)
        seg = self.dec2(seg, x1)
        seg = self.dec1(seg, x0)
        seg = self.seg_final_up(seg)
        seg_out = self.seg_head(seg)

        # ========== 分类分支 (带ROI Masking) ==========
        if self.use_attention:
            cls_feat = self.attention(x4)
        else:
            cls_feat = x4

        # 使用ROI Masking: 只在胰腺区域内池化
        if self.use_roi_masking and pancreas_mask is not None:
            cls_feat = self.masked_global_average_pool(cls_feat, pancreas_mask)
        else:
            # 标准全局平均池化
            cls_feat = self.cls_gap(cls_feat)
            cls_feat = cls_feat.view(cls_feat.size(0), -1)

        cls_main = self.cls_head(cls_feat)

        # 辅助分类 (也使用ROI Masking)
        if self.use_roi_masking and pancreas_mask is not None:
            # 对seg特征也应用mask
            seg_pooled = self.masked_global_average_pool(seg, pancreas_mask)
            seg_aux = self.seg_aux_cls(seg_pooled.unsqueeze(-1).unsqueeze(-1))
        else:
            seg_aux = self.seg_aux_cls(seg)

        # 融合
        weight = F.softmax(self.fusion_weight, dim=0)
        cls_out = weight[0] * cls_main + weight[1] * seg_aux

        return {
            'segmentation': seg_out,      # (B, 1, H, W)
            'classification': cls_out,     # (B, num_cls_classes)
            'seg_features': seg,
            'cls_main': cls_main,
            'seg_aux': seg_aux
        }


class ResNet253DMultiTask(nn.Module):
    """
    真3D版本: 处理整个3D volume
    通过3D卷积或逐slice处理+3D池化

    预训练权重支持:
        - pretrained=True: 使用ImageNet预训练权重
        - pretrained=False: 不使用预训练权重
        - pretrained="path/to/weights.pth": 加载自定义权重文件
    """

    def __init__(self, num_seg_classes=1, num_cls_classes=2,
                 depth=32, use_attention=True, pretrained=True, dropout=0.3):
        super().__init__()

        self.depth = depth

        # 使用2D ResNet处理每个slice，然后在深度方向聚合
        if isinstance(pretrained, str):
            # 加载自定义权重文件
            backbone = resnet18(weights=None)
            self._load_custom_weights(backbone, pretrained)
            use_pretrained_conv = True
        elif pretrained:
            # 使用ImageNet预训练权重
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
            use_pretrained_conv = True
        else:
            # 不使用预训练权重
            backbone = resnet18(weights=None)
            use_pretrained_conv = False

        # 第一层改为单通道
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if use_pretrained_conv:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 3D聚合模块
        self.temporal_conv = nn.Conv3d(512, 512, kernel_size=(3, 1, 1),
                                        padding=(1, 0, 0), bias=False)
        self.temporal_bn = nn.BatchNorm3d(512)

        # 3D时序聚合用于分割
        self.seg_temporal_conv = nn.Conv3d(512, 512, kernel_size=(3, 1, 1),
                                           padding=(1, 0, 0), bias=False)
        self.seg_temporal_bn = nn.BatchNorm3d(512)

        # 分割解码器 - 逐slice解码后保持3D结构
        self.seg_decoder = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64)
        ])
        self.seg_head = SegmentationHead(64, num_seg_classes)

        # 分类头
        self.cls_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_cls_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W)
        """
        b, c, d, h, w = x.shape

        # 将D维度合并到batch: (B*D, 1, H, W)
        x = x.view(b * d, c, h, w)

        # 编码
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0_pool = self.maxpool(x0)
        x1 = self.layer1(x0_pool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # (B*D, 512, H/32, W/32)

        # 恢复D维度: (B, 512, D, H/32, W/32)
        _, c, h_out, w_out = x4.shape
        x4 = x4.view(b, d, c, h_out, w_out).permute(0, 2, 1, 3, 4)

        # 分类: 使用3D时序卷积聚合
        cls_feat = self.temporal_conv(x4)
        cls_feat = self.temporal_bn(cls_feat)
        cls_feat = F.relu(cls_feat)

        # 分类: 3D池化
        cls_pooled = self.cls_pool(cls_feat).view(b, -1)
        cls_out = self.cls_head(cls_pooled)

        # 分割: 独立的3D时序聚合路径
        seg_feat = self.seg_temporal_conv(x4)
        seg_feat = self.seg_temporal_bn(seg_feat)
        seg_feat = F.relu(seg_feat)

        # 分割: 将(B, C, D, H, W)转换为(B*D, C, H, W)进行逐slice解码
        # 或者保持3D结构，但使用2D解码器逐slice处理
        # 这里采用：对每个深度位置独立解码，然后重组为3D

        b, c, d_out, h_out, w_out = seg_feat.shape
        # (B, C, D, H, W) -> (B*D, C, H, W)
        seg_feat_2d = seg_feat.permute(0, 2, 1, 3, 4).contiguous().view(b * d_out, c, h_out, w_out)

        # 解码: 每个slice独立通过解码器
        for decoder in self.seg_decoder:
            seg_feat_2d = decoder(seg_feat_2d)

        # 分割头
        seg_out_2d = self.seg_head(seg_feat_2d)  # (B*D, 1, H, W)

        # 重组回3D: (B*D, 1, H, W) -> (B, 1, D, H, W)
        seg_out = seg_out_2d.view(b, d_out, 1, seg_out_2d.shape[-2], seg_out_2d.shape[-1]).permute(0, 2, 1, 3, 4)

        return {
            'segmentation': seg_out,
            'classification': cls_out
        }

    def _load_custom_weights(self, model, weights_path):
        """加载自定义预训练权重"""
        print(f"Loading custom pretrained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

        # 处理不同格式的权重文件
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 移除可能的'module.'前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Custom weights loaded successfully")


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self, seg_weight=1.0, cls_weight=1.0,
                 use_dice=True, use_focal=False, cls_weights=None):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.use_dice = use_dice

        self.seg_bce = nn.BCEWithLogitsLoss()

        if use_focal:
            # 对于2.4:1的不平衡，设置alpha > 0.5来增加少数类权重
            # alpha=0.7表示正类(少数类)权重0.7，负类权重0.3
            self.cls_criterion = FocalLoss(alpha=0.7, gamma=2.0, cls_weights=cls_weights)
        else:
            # 使用类别权重来处理不平衡
            self.cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, predictions, targets):
        seg_pred = predictions['segmentation']
        cls_pred = predictions['classification']

        pancreas_mask = targets['pancreas_mask']
        gata6_label = targets['gata6_label']

        # 分割损失
        seg_loss = self.seg_bce(seg_pred, pancreas_mask)
        if self.use_dice:
            seg_loss = seg_loss + self.dice_loss(seg_pred, pancreas_mask)

        # 分类损失
        cls_loss = self.cls_criterion(cls_pred, gata6_label)

        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        return {
            'total': total_loss,
            'seg': seg_loss,
            'cls': cls_loss
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    alpha: 正类权重系数 (对于正类是少数类的情况, 应设置alpha > 0.5, 如0.7)
           alpha越大，正类权重越高
    gamma: 聚焦参数, 减少对易分类样本的关注
    cls_weights: 类别权重tensor (用于CrossEntropy的weight参数)
    """
    def __init__(self, alpha=0.3, gamma=2.0, reduction='mean', cls_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cls_weights = cls_weights
        # 预注册buffer，避免每次forward创建新tensor
        # alpha_pos是正类(标签=1)的权重，alpha_neg是负类(标签=0)的权重
        self.register_buffer('alpha_pos', torch.tensor(alpha))
        self.register_buffer('alpha_neg', torch.tensor(1 - alpha))

    def forward(self, inputs, targets):
        # 计算交叉熵 (使用类别权重)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.cls_weights, reduction='none')

        # 数值稳定性：防止pt过小时exp下溢
        # 使用clamp限制ce_loss范围，防止pt接近0或1时的数值问题
        ce_loss_clamped = ce_loss.clamp(min=1e-7, max=50.0)
        pt = torch.exp(-ce_loss_clamped)

        # 进一步限制pt范围，防止(1-pt)出现数值问题
        pt = pt.clamp(min=1e-7, max=1.0 - 1e-7)

        # 调整alpha: 对正类样本(假设为少数类1)给予更高权重
        # alpha_t = alpha for positive class, 1-alpha for negative class
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)

        focal_term = (1 - pt) ** self.gamma
        loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PredictionsAggregator:
    """
    聚合2.5D模型在重叠slice上的预测结果
    """

    def __init__(self, depth, slices_per_input, stride=1):
        self.depth = depth
        self.slices_per_input = slices_per_input
        self.stride = stride

    def aggregate_classification(self, predictions, weights=None):
        """
        聚合分类预测
        Args:
            predictions: list of (num_classes,) arrays
            weights: 可选的权重
        Returns:
            聚合后的概率分布
        """
        predictions = np.array(predictions)  # (N, num_classes)

        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()
            avg_pred = np.average(predictions, axis=0, weights=weights)
        else:
            avg_pred = predictions.mean(axis=0)

        return avg_pred

    def aggregate_segmentation(self, seg_outputs, slice_indices, output_shape):
        """
        聚合分割预测 (加权平均重叠区域)
        注意：模型只预测中间slice的分割结果
        Args:
            seg_outputs: list of (H, W) segmentation predictions (每个是中间slice的预测)
            slice_indices: list of slice index tuples (如 [0,1,2], [1,2,3] 等)
            output_shape: (D, H, W)
        Returns:
            聚合后的3D分割结果
        """
        D, H, W = output_shape
        aggregated = np.zeros((D, H, W), dtype=np.float32)
        weight_map = np.zeros((D, H, W), dtype=np.float32)

        for seg, indices in zip(seg_outputs, slice_indices):
            # 模型只预测中间slice，所以只分配到中间位置
            mid_idx = len(indices) // 2
            target_slice = indices[mid_idx]
            if 0 <= target_slice < D:
                aggregated[target_slice] += seg
                weight_map[target_slice] += 1

        # 对于没有被预测的slice，使用邻近值插值或保持为0
        # 归一化
        weight_map = np.maximum(weight_map, 1e-5)
        aggregated = aggregated / weight_map

        return aggregated


def test_models():
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    h, w = 256, 256

    print("="*60)
    print("Testing ResNet25DMultiTask (2.5D)")
    print("="*60)

    # 测试2.5D模型
    model_25d = ResNet25DMultiTask(
        num_seg_classes=1,
        num_cls_classes=2,
        slices_per_input=3,
        use_attention=True,
        pretrained=True
    ).to(device)

    x_25d = torch.randn(batch_size, 3, h, w).to(device)
    with torch.no_grad():
        out_25d = model_25d(x_25d)

    print(f"Input shape: {x_25d.shape}")
    for key, val in out_25d.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    total_params = sum(p.numel() for p in model_25d.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "="*60)
    print("Testing ResNet253DMultiTask (3D)")
    print("="*60)

    # 测试3D模型
    model_3d = ResNet253DMultiTask(
        num_seg_classes=1,
        num_cls_classes=2,
        depth=32,
        use_attention=True,
        pretrained=True
    ).to(device)

    x_3d = torch.randn(batch_size, 1, 32, h, w).to(device)
    with torch.no_grad():
        out_3d = model_3d(x_3d)

    print(f"Input shape: {x_3d.shape}")
    for key, val in out_3d.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    total_params = sum(p.numel() for p in model_3d.parameters())
    print(f"Total parameters: {total_params:,}")


if __name__ == '__main__':
    test_models()
