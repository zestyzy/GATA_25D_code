"""
简化版推理脚本 - 快速对单个患者进行预测
"""
import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from model import MultiTaskResNet18


def load_model(checkpoint_path, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get('args', {})

    model = MultiTaskResNet18(
        num_seg_classes=1,
        num_cls_classes=2,
        use_attention=args_dict.get('use_attention', True),
        pretrained=False,
        dropout=args_dict.get('dropout', 0.3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


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

    return img_tensor


def predict(model, image_path, device='cpu'):
    """
    对nii图像进行预测

    Returns:
        dict: 包含分类预测结果和分割概率图
    """
    # 加载图像
    sitk_image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(sitk_image)

    if len(image.shape) == 4:
        image = image[:, :, :, 0]

    print(f"Image shape: {image.shape}")

    # 处理每个切片
    all_cls_probs = []
    all_seg_probs = []

    with torch.no_grad():
        for z in range(image.shape[0]):
            img_slice = image[z, :, :].astype(np.float32)
            img_tensor = preprocess_slice(img_slice, target_size=256)
            img_tensor = img_tensor.to(device)

            # 推理
            outputs = model(img_tensor)

            cls_prob = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
            seg_prob = torch.sigmoid(outputs['segmentation']).cpu().numpy()[0, 0]

            all_cls_probs.append(cls_prob)
            all_seg_probs.append(seg_prob)

    # 聚合结果
    avg_cls_prob = np.mean(all_cls_probs, axis=0)
    cls_pred = int(avg_cls_prob.argmax())
    cls_confidence = float(avg_cls_prob.max())

    # 分割体数据
    seg_volume = np.stack(all_seg_probs, axis=0)

    return {
        'gata6_prediction': cls_pred,  # 0 或 1
        'gata6_confidence': cls_confidence,
        'gata6_probability_negative': float(avg_cls_prob[0]),
        'gata6_probability_positive': float(avg_cls_prob[1]),
        'segmentation_volume': seg_volume,
        'num_slices': image.shape[0]
    }


def main():
    parser = argparse.ArgumentParser(description='Predict GATA6 and pancreas segmentation')
    parser.add_argument('--image', type=str, required=True, help='Path to image.nii file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_seg', type=str, help='Output path for segmentation (optional)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # 预测
    print(f"\nProcessing: {args.image}")
    results = predict(model, args.image, device)

    # 输出结果
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"GATA6 Mutation Prediction: {'POSITIVE (1)' if results['gata6_prediction'] == 1 else 'NEGATIVE (0)'}")
    print(f"  Confidence: {results['gata6_confidence']:.4f}")
    print(f"  Probability (Negative): {results['gata6_probability_negative']:.4f}")
    print(f"  Probability (Positive): {results['gata6_probability_positive']:.4f}")
    print(f"\nSegmentation: Generated for {results['num_slices']} slices")
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
