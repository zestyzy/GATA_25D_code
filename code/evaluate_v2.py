"""
2.5D/3D多任务评估脚本
支持:
1. 对整个测试集进行评估
2. 对单个患者进行推理
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import SimpleITK as sitk
from tqdm import tqdm

from dataset_v2 import PancreasCaseDataset
from model_v2 import ResNet25DMultiTask, ResNet253DMultiTask, PredictionsAggregator


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get('args', {})

    model_type = args_dict.get('model_type', '2.5d')

    if model_type == '2.5d':
        model = ResNet25DMultiTask(
            num_seg_classes=1,
            num_cls_classes=2,
            slices_per_input=args_dict.get('slices_per_input', 3),
            use_attention=args_dict.get('use_attention', True),
            pretrained=False,
            dropout=args_dict.get('dropout', 0.3),
            use_roi_masking=args_dict.get('use_roi_masking', True)
        )
    else:
        model = ResNet253DMultiTask(
            num_seg_classes=1,
            num_cls_classes=2,
            depth=args_dict.get('depth', 32),
            use_attention=args_dict.get('use_attention', True),
            pretrained=False,
            dropout=args_dict.get('dropout', 0.3)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint, args_dict


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_type='2.5d', args=None):
    """评估整个数据集，支持case级别聚合"""
    from model_v2 import PredictionsAggregator

    model.eval()

    # case级别聚合
    case_predictions = {}

    print("Evaluating...")
    for batch in tqdm(dataloader):
        if model_type == 'case':
            images = batch['image'].to(device)
            mid_d = images.shape[2] // 2
            pancreas_masks = batch['pancreas_mask'][:, 0, mid_d, :, :].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label']
            case_ids = batch['case_id']
            slice_indices = None
        else:
            images = batch['image'].to(device)
            pancreas_masks = batch['pancreas_mask'].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label']
            case_ids = batch['case_id']
            slice_indices = batch.get('slice_indices', None)

        # 前向传播 (使用ROI Masking)
        outputs = model(images, pancreas_masks)

        cls_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()
        seg_probs = torch.sigmoid(outputs['segmentation']).cpu().numpy()

        # 收集每个case的预测
        for i, case_id in enumerate(case_ids):
            case_id = int(case_id)
            if case_id not in case_predictions:
                case_predictions[case_id] = {
                    'cls_probs': [],
                    'seg_outputs': [],
                    'slice_indices': [],
                    'label': int(gata6_labels[i].item()),
                    'masks_3d': []  # 收集3D masks用于Dice计算
                }
            case_predictions[case_id]['cls_probs'].append(cls_probs[i])
            case_predictions[case_id]['seg_outputs'].append(seg_probs[i, 0])

            if slice_indices is not None and len(slice_indices) > i:
                case_predictions[case_id]['slice_indices'].append(slice_indices[i].tolist())

            # 收集3D mask (只在验证时使用第一个batch的mask避免重复)
            if 'pancreas_mask_3d' in batch and len(case_predictions[case_id]['masks_3d']) == 0:
                case_predictions[case_id]['masks_3d'] = batch['pancreas_mask_3d'][i].numpy()

    # ========== 聚合case级别预测 ==========
    all_preds = []
    all_labels = []
    all_probs = []
    all_case_ids = []
    seg_dice_scores = []

    depth = args.depth if args else 32
    image_size = args.image_size if args else 256

    for case_id, case_data in case_predictions.items():
        # 聚合分类预测
        cls_probs = np.array(case_data['cls_probs'])
        avg_cls_prob = cls_probs.mean(axis=0)
        pred_prob = float(avg_cls_prob[1])
        # 使用与训练/验证一致的阈值0.4
        pred_label = 1 if pred_prob >= 0.4 else 0

        all_preds.append(pred_label)
        all_labels.append(case_data['label'])
        all_probs.append(pred_prob)
        all_case_ids.append(case_id)

        # 聚合分割预测
        if len(case_data['slice_indices']) > 0 and len(case_data['seg_outputs']) > 0:
            aggregator = PredictionsAggregator(
                depth=depth,
                slices_per_input=len(case_data['slice_indices'][0]),
                stride=1
            )
            aggregated_seg = aggregator.aggregate_segmentation(
                case_data['seg_outputs'],
                case_data['slice_indices'],
                (depth, image_size, image_size)
            )

            # 计算2D Dice (只在有胰腺GT的slice上计算，与train_v2.py一致)
            if len(case_data['masks_3d']) > 0:
                gt_mask = case_data['masks_3d']
                if gt_mask.shape == aggregated_seg.shape:
                    # 逐slice计算2D Dice，只计算有胰腺GT的slice
                    slice_dice_scores = []
                    for d_idx in range(gt_mask.shape[0]):
                        gt_slice = gt_mask[d_idx]
                        pred_slice = aggregated_seg[d_idx]
                        # 只计算有胰腺GT的slice
                        if gt_slice.max() > 0:
                            dice_2d = compute_2d_dice(pred_slice, gt_slice)
                            slice_dice_scores.append(dice_2d)
                    # 使用平均2D Dice
                    if len(slice_dice_scores) > 0:
                        dice = np.mean(slice_dice_scores)
                        seg_dice_scores.append(dice)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'mean_dice': np.mean(seg_dice_scores) if seg_dice_scores else 0.0,
        'std_dice': np.std(seg_dice_scores) if seg_dice_scores else 0.0,
        'num_cases_evaluated': len(seg_dice_scores)
    }

    return metrics, all_preds, all_labels, all_probs, all_case_ids


def compute_dice(pred, target, smooth=1e-5):
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def compute_dice_volume(pred, target, smooth=1e-5):
    """
    计算3D volume的Dice系数
    Args:
        pred: (D, H, W) 预测mask
        target: (D, H, W) 真实mask
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def compute_2d_dice(pred, target, smooth=1e-5):
    """
    计算2D Dice系数
    Args:
        pred: (H, W) 预测概率或二值化预测
        target: (H, W) 二值化真实标签
    """
    # 二值化预测
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


@torch.no_grad()
def predict_case(model, image_path, organ_mask_path, device, args):
    """对单个患者进行推理，使用聚合器处理重叠预测，支持ROI Masking"""
    model.eval()

    sitk_image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(sitk_image)
    sitk_mask = sitk.ReadImage(organ_mask_path)
    organ_mask = sitk.GetArrayFromImage(sitk_mask)

    if len(image.shape) == 4:
        image = image[:, :, :, 0]

    # 多器官ROI提取
    multi_organ = (organ_mask == 1) | (organ_mask == 2) | (organ_mask == 3)
    if not np.any(multi_organ):
        d, h, w = image.shape
        bbox = (d//4, 3*d//4, h//4, 3*h//4, w//4, 3*w//4)
    else:
        z_idx, y_idx, x_idx = np.where(multi_organ)
        bbox = (z_idx.min(), z_idx.max(), y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max())

    z_min, z_max = bbox[0], bbox[1]
    z_start = max(0, z_min - args.z_extension)
    z_end = min(image.shape[0], z_max + args.z_extension + 1)

    image_crop = image[z_start:z_end, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1]
    # 提取胰腺mask用于ROI Masking
    pancreas_mask_crop = (organ_mask[z_start:z_end, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1] == 1).astype(np.float32)

    # 归一化
    ww, wl = 400, 40
    min_val, max_val = wl - ww//2, wl + ww//2
    image_crop = np.clip(image_crop, min_val, max_val)
    image_crop = (image_crop - min_val) / (max_val - min_val)

    # 重采样
    import torch.nn.functional as F
    image_tensor = torch.from_numpy(image_crop).float().unsqueeze(0).unsqueeze(0)
    pancreas_mask_tensor = torch.from_numpy(pancreas_mask_crop).float().unsqueeze(0).unsqueeze(0)

    target_d, target_h, target_w = args.depth, args.image_size, args.image_size
    image_resampled = F.interpolate(image_tensor, size=(target_d, target_h, target_w),
                                     mode='trilinear', align_corners=False)
    # 对mask使用最近邻插值保持二值性
    pancreas_mask_resampled = F.interpolate(pancreas_mask_tensor, size=(target_d, target_h, target_w),
                                             mode='nearest')

    # 创建聚合器
    aggregator = PredictionsAggregator(
        depth=target_d,
        slices_per_input=args.slices_per_input,
        stride=1
    )

    # 2.5D推理 - 收集所有slice group的预测
    all_cls_probs = []
    seg_outputs = []
    slice_indices_list = []

    for d in range(0, target_d - args.slices_per_input + 1):
        indices = list(range(d, d + args.slices_per_input))
        input_25d = image_resampled[0, 0, d:d+args.slices_per_input, :, :].unsqueeze(0).to(device)
        # 提取对应的mask (取中间slice)
        mid_idx = d + args.slices_per_input // 2
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

    return {
        'classification': {
            'pred': int(aggregated_cls.argmax()),
            'confidence': float(aggregated_cls.max()),
            'probabilities': aggregated_cls.tolist()
        },
        'segmentation': aggregated_seg,  # (D, H, W)
        'bbox': bbox,
        'num_predictions': len(all_cls_probs)
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint, model_args = load_model(args.checkpoint, device)
    print(f"Model loaded (type: {model_args.get('model_type', '2.5d')})")

    if args.mode == 'dataset':
        print("\nEvaluating dataset...")

        dataset = PancreasCaseDataset(
            data_root=args.data_root,
            label_file=args.label_file,
            target_size=(args.image_size, args.image_size, args.depth),
            z_extension=args.z_extension,
            normalize_mode='window',
            transform=None
        )

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        metrics, preds, labels, probs, case_ids = evaluate_dataset(
            model, dataloader, device, model_args.get('train_mode', 'slice'), args
        )

        print("\n" + "="*50)
        print("Classification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"\nConfusion Matrix: {metrics['confusion_matrix']}")

        print("\nSegmentation Metrics (3D Volume):")
        print(f"  Mean Dice: {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f}")
        print(f"  Cases Evaluated: {metrics['num_cases_evaluated']}")
        print("="*50)

        results = {
            'metrics': metrics,
            'predictions': [
                {'case_id': int(cid), 'pred': int(p), 'label': int(l), 'prob': float(pr)}
                for cid, p, l, pr in zip(case_ids, preds, labels, probs)
            ]
        }

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    elif args.mode == 'patient':
        print(f"\nPredicting: {args.patient_dir}")

        image_path = os.path.join(args.patient_dir, 'image.nii')
        organ_mask_path = os.path.join(args.patient_dir, 'organ_label.nii')

        results = predict_case(model, image_path, organ_mask_path, device, args)

        print("\n" + "="*50)
        print("Classification:")
        print(f"  Prediction: {results['classification']['pred']}")
        print(f"  Confidence: {results['classification']['confidence']:.4f}")
        print(f"  Aggregated from {results['num_predictions']} overlapping predictions")
        print(f"  Probabilities: {results['classification']['probabilities']}")
        print("="*50)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'prediction.json'), 'w') as f:
            json.dump(results['classification'], f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 2.5D/3D Model')

    parser.add_argument('--mode', type=str, choices=['dataset', 'patient'], default='dataset')
    parser.add_argument('--checkpoint', type=str, required=True)

    parser.add_argument('--data_root', type=str,
                        default='/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/zs')
    parser.add_argument('--label_file', type=str,
                        default='/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/GATA6_label.xlsx')
    parser.add_argument('--patient_dir', type=str)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=32)
    parser.add_argument('--z_extension', type=int, default=3)
    parser.add_argument('--slices_per_input', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_v2')

    args = parser.parse_args()
    main(args)
