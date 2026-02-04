"""
2.5D/3D多任务训练脚本
胰腺分割 + GATA6基因二分类
支持Case级别和Slice级别两种训练模式
"""
import os
import sys
import argparse
import json
import time
import random
import csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm
import SimpleITK as sitk

from dataset_v2 import PancreasCaseDataset, Slice2p5DDataset, get_transforms
from model_v2 import ResNet25DMultiTask, ResNet253DMultiTask, MultiTaskLoss


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def get_dataloaders(args):
    """创建数据加载器"""
    print("Loading dataset...")

    # 创建完整case数据集
    full_dataset = PancreasCaseDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        target_size=(args.image_size, args.image_size, args.depth),
        z_extension=args.z_extension,
        normalize_mode=args.normalize_mode,
        transform=None  # 变换在2.5D数据集或训练循环中应用
    )

    # 分层划分训练集和验证集
    all_indices = list(range(len(full_dataset)))
    patient_labels = [full_dataset.samples[i]['gata6_label'] for i in all_indices]

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=patient_labels
    )

    print(f"Train cases: {len(train_indices)}, Val cases: {len(val_indices)}")

    # 根据训练模式创建数据集
    if args.train_mode == 'case':
        # Case级别: 每个样本是一个完整3D volume
        from torch.utils.data import Subset

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    else:  # slice模式
        # 创建训练集和验证集的base dataset
        train_base = PancreasCaseDataset(
            data_root=args.data_root,
            label_file=args.label_file,
            target_size=(args.image_size, args.image_size, args.depth),
            z_extension=args.z_extension,
            normalize_mode=args.normalize_mode,
            transform=get_transforms('train')
        )
        train_base.samples = [train_base.samples[i] for i in train_indices]

        val_base = PancreasCaseDataset(
            data_root=args.data_root,
            label_file=args.label_file,
            target_size=(args.image_size, args.image_size, args.depth),
            z_extension=args.z_extension,
            normalize_mode=args.normalize_mode,
            transform=None
        )
        val_base.samples = [val_base.samples[i] for i in val_indices]

        # 创建2.5D数据集 - 使用stride=1滑动窗口生成重叠样本
        train_dataset = Slice2p5DDataset(
            train_base,
            slices_per_input=args.slices_per_input,
            stride=args.stride
        )
        val_dataset = Slice2p5DDataset(
            val_base,
            slices_per_input=args.slices_per_input,
            stride=args.stride
        )

        print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

        # 使用shuffle=True (移除WeightedRandomSampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch，带tqdm进度条
    分类指标同时计算case-level和slice-level，明确区分统计口径
    """
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_cls_loss = 0

    # case级别聚合
    case_predictions = {}  # case_id -> {'cls_probs': [], 'seg_preds': [], 'seg_gts': [], 'label': int}
    # slice级别统计（用于对比，避免重复标签导致的虚高）
    slice_preds = []
    slice_labels = []
    # 训练分割Dice统计
    batch_dice_scores = []

    # 使用tqdm包装dataloader
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]', leave=False)

    for batch_idx, batch in enumerate(pbar):
        # 根据训练模式处理输入
        if args.train_mode == 'case':
            # 3D输入: (B, 1, D, H, W)
            images = batch['image'].to(device)
            # 取中间slice作为分割目标
            mid_d = images.shape[2] // 2
            pancreas_masks = batch['pancreas_mask'][:, 0, mid_d, :, :].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label'].to(device)
            case_ids = batch['case_id']
        else:
            # 2.5D输入: (B, slices_per_input, H, W)
            images = batch['image'].to(device)
            pancreas_masks = batch['pancreas_mask'].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label'].to(device)
            case_ids = batch['case_id']

        optimizer.zero_grad()

        # 前向传播 (传入pancreas_mask用于ROI Masking)
        outputs = model(images, pancreas_masks)

        # 计算损失
        targets = {
            'pancreas_mask': pancreas_masks,
            'gata6_label': gata6_labels
        }
        losses = criterion(outputs, targets)

        # 反向传播
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += losses['total'].item()
        total_seg_loss += losses['seg'].item()
        total_cls_loss += losses['cls'].item()

        # 收集分类预测 (按case聚合 + slice级别统计)
        cls_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()
        pos_probs = cls_probs[:, 1]  # 正类概率
        # slice-level预测（使用0.4阈值）
        batch_slice_preds = (pos_probs >= 0.4).astype(int)
        slice_preds.extend(batch_slice_preds)
        slice_labels.extend(gata6_labels.cpu().numpy())

        # 收集分割预测用于计算训练Dice
        seg_preds = torch.sigmoid(outputs['segmentation']).detach().cpu().numpy()  # (B, 1, H, W)
        seg_gts = pancreas_masks.cpu().numpy()  # (B, 1, H, W)

        for i, case_id in enumerate(case_ids):
            case_id = int(case_id)
            if case_id not in case_predictions:
                case_predictions[case_id] = {
                    'cls_probs': [],
                    'seg_preds': [],  # 收集分割预测
                    'seg_gts': [],    # 收集分割GT
                    'label': int(gata6_labels[i].item())
                }
            case_predictions[case_id]['cls_probs'].append(pos_probs[i])
            case_predictions[case_id]['seg_preds'].append(seg_preds[i, 0])  # (H, W)
            case_predictions[case_id]['seg_gts'].append(seg_gts[i, 0])       # (H, W)

        # 计算当前batch的2D Dice（用于实时监控）
        with torch.no_grad():
            seg_pred_binary = (torch.sigmoid(outputs['segmentation']) > 0.5).float()
            seg_gt_binary = pancreas_masks
            intersection = (seg_pred_binary * seg_gt_binary).sum(dim=(1, 2, 3))
            union = seg_pred_binary.sum(dim=(1, 2, 3)) + seg_gt_binary.sum(dim=(1, 2, 3))
            dice_batch = (2. * intersection + 1e-5) / (union + 1e-5)
            batch_dice_scores.extend(dice_batch.cpu().numpy())

        # 更新进度条显示
        current_loss = total_loss / (batch_idx + 1)
        current_seg = total_seg_loss / (batch_idx + 1)
        current_cls = total_cls_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'seg': f'{current_seg:.4f}',
            'cls': f'{current_cls:.4f}'
        })

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_seg_loss = total_seg_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches

    # 计算slice-level指标（反映重复标签的影响）
    slice_acc = accuracy_score(slice_labels, slice_preds)
    slice_f1 = f1_score(slice_labels, slice_preds, zero_division=0)

    # 按case聚合计算指标 (真实case-level性能)
    all_preds = []
    all_labels = []
    all_probs = []

    # 训练分割Dice计算
    case_dice_scores = []
    for case_id, case_data in case_predictions.items():
        # 聚合分类预测 (取平均)
        case_probs = np.array(case_data['cls_probs'])
        avg_prob = case_probs.mean()

        # 使用0.4阈值
        pred_label = 1 if avg_prob >= 0.4 else 0

        all_preds.append(pred_label)
        all_labels.append(case_data['label'])
        all_probs.append(avg_prob)

        # 计算case-level平均2D Dice
        if len(case_data['seg_preds']) > 0:
            case_dice_list = []
            for seg_pred, seg_gt in zip(case_data['seg_preds'], case_data['seg_gts']):
                # 只计算有GT的slice
                if seg_gt.max() > 0:
                    dice_2d = compute_2d_dice(seg_pred, seg_gt)
                    case_dice_list.append(dice_2d)
            if len(case_dice_list) > 0:
                case_dice = np.mean(case_dice_list)
                case_dice_scores.append(case_dice)

    # 计算case-level指标
    cls_acc = accuracy_score(all_labels, all_preds)
    cls_f1 = f1_score(all_labels, all_preds, zero_division=0)
    cls_precision = precision_score(all_labels, all_preds, zero_division=0)
    cls_recall = recall_score(all_labels, all_preds, zero_division=0)

    # 计算训练分割Dice
    train_seg_dice = np.mean(case_dice_scores) if case_dice_scores else 0.0
    batch_avg_dice = np.mean(batch_dice_scores) if batch_dice_scores else 0.0

    # 打印训练分类指标（区分case-level和slice-level）
    if args.train_mode == 'slice':
        # 2.5D模式：显示两种口径的对比
        print(f"  [Train Slice-level] Acc: {slice_acc:.4f}, F1: {slice_f1:.4f} (重复标签，参考用)")
        print(f"  [Train Case-level]  Acc: {cls_acc:.4f}, Precision: {cls_precision:.4f}, "
              f"Recall: {cls_recall:.4f}, F1: {cls_f1:.4f} (真实性能)")
    else:
        # case模式：只有case-level
        print(f"  [Train] Acc: {cls_acc:.4f}, Precision: {cls_precision:.4f}, "
              f"Recall: {cls_recall:.4f}, F1: {cls_f1:.4f}")
    # 打印分割Dice
    print(f"  [Train Segmentation] Case-Dice: {train_seg_dice:.4f}, Batch-Dice: {batch_avg_dice:.4f}")
    print(f"  [Train Distribution] Pred: Pos={sum(all_preds)}, Neg={len(all_preds)-sum(all_preds)} | "
          f"True: Pos={sum(all_labels)}, Neg={len(all_labels)-sum(all_labels)}")

    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'cls_loss': avg_cls_loss,
        'cls_acc': cls_acc,  # case-level
        'cls_f1': cls_f1,    # case-level
        'seg_dice': train_seg_dice,  # 训练分割Dice
        'slice_acc': slice_acc if args.train_mode == 'slice' else cls_acc,
        'slice_f1': slice_f1 if args.train_mode == 'slice' else cls_f1
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, args, cls_threshold=None, epoch=0, output_dir=None):
    """验证，带tqdm进度条，支持case级别聚合评估和可视化
    Args:
        cls_threshold: 分类阈值，None则使用0.4
        epoch: 当前epoch数，用于可视化文件夹命名
        output_dir: 输出目录，用于保存可视化结果
    """
    from model_v2 import PredictionsAggregator
    # SimpleITK和Path已在文件顶部导入

    # 对于不平衡数据，使用较低阈值来提升召回率
    if cls_threshold is None:
        cls_threshold = 0.4

    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_cls_loss = 0

    # case级别聚合 - 同时存储真实mask用于Dice计算
    case_predictions = {}  # case_id -> {'cls_probs': [], 'seg_outputs': [], 'slice_indices': [], 'label': int, 'masks': []}

    # 使用tqdm包装dataloader
    pbar = tqdm(dataloader, desc='[Val]', leave=False)

    for batch_idx, batch in enumerate(pbar):
        # 根据训练模式处理输入
        if args.train_mode == 'case':
            images = batch['image'].to(device)
            mid_d = images.shape[2] // 2
            pancreas_masks = batch['pancreas_mask'][:, 0, mid_d, :, :].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label'].to(device)
            case_ids = batch['case_id']
            slice_indices = None
            # 获取完整的3D mask用于后续Dice计算
            pancreas_masks_3d = batch['pancreas_mask'].cpu().numpy()  # (B, 1, D, H, W)
        else:
            images = batch['image'].to(device)
            pancreas_masks = batch['pancreas_mask'].unsqueeze(1).to(device)
            gata6_labels = batch['gata6_label'].to(device)
            case_ids = batch['case_id']
            slice_indices = batch.get('slice_indices', None)
            # 获取完整的3D mask (如果存在)
            if 'pancreas_mask_3d' in batch:
                pancreas_masks_3d = batch['pancreas_mask_3d'].cpu().numpy()  # (B, D, H, W)
            else:
                pancreas_masks_3d = None

        # 前向传播 (传入pancreas_mask用于ROI Masking)
        outputs = model(images, pancreas_masks)

        # 计算损失
        targets = {
            'pancreas_mask': pancreas_masks,
            'gata6_label': gata6_labels
        }
        losses = criterion(outputs, targets)

        total_loss += losses['total'].item()
        total_seg_loss += losses['seg'].item()
        total_cls_loss += losses['cls'].item()

        # 收集每个case的预测用于聚合
        cls_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()
        seg_outputs = torch.sigmoid(outputs['segmentation']).cpu().numpy()

        for i, case_id in enumerate(case_ids):
            case_id = int(case_id)
            if case_id not in case_predictions:
                case_predictions[case_id] = {
                    'cls_probs': [],
                    'seg_outputs': [],
                    'slice_indices': [],
                    'label': int(gata6_labels[i].item()),
                    'masks': []  # 存储真实mask
                }
            case_predictions[case_id]['cls_probs'].append(cls_probs[i])
            case_predictions[case_id]['seg_outputs'].append(seg_outputs[i, 0])

            # 存储真实mask用于Dice计算（每个case只存储一次）
            if pancreas_masks_3d is not None and len(case_predictions[case_id]['masks']) == 0:
                mask_data = pancreas_masks_3d[i]
                # 如果是3D mask (D, H, W) 或 (1, D, H, W)
                if len(mask_data.shape) == 3:
                    case_predictions[case_id]['masks'].append(mask_data)
                elif len(mask_data.shape) == 4:
                    case_predictions[case_id]['masks'].append(mask_data[0])

            if slice_indices is not None and len(slice_indices) > i:
                case_predictions[case_id]['slice_indices'].append(slice_indices[i].tolist())

        # 更新进度条
        current_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})

    # ========== 聚合case级别预测并计算Dice ==========
    all_preds = []
    all_labels = []
    all_probs = []
    seg_dice_scores = []

    # 随机选择一个case进行可视化
    import random
    vis_case_id = None
    vis_data = None
    case_ids_list = list(case_predictions.keys())
    selected_case_for_vis = random.choice(case_ids_list) if len(case_ids_list) > 0 else None

    depth = args.depth

    for idx, (case_id, case_data) in enumerate(case_predictions.items()):
        # 聚合分类预测
        cls_probs = np.array(case_data['cls_probs'])
        avg_cls_prob = cls_probs.mean(axis=0)

        # 使用阈值而不是argmax
        pred_prob = float(avg_cls_prob[1])
        pred_label = 1 if pred_prob >= cls_threshold else 0

        all_preds.append(pred_label)
        all_labels.append(case_data['label'])
        all_probs.append(pred_prob)

        # 聚合分割预测
        if args.train_mode == 'slice' and len(case_data['slice_indices']) > 0:
            aggregator = PredictionsAggregator(
                depth=depth,
                slices_per_input=args.slices_per_input,
                stride=args.stride
            )
            aggregated_seg = aggregator.aggregate_segmentation(
                case_data['seg_outputs'],
                case_data['slice_indices'],
                (depth, args.image_size, args.image_size)
            )  # (D, H, W)
        else:
            seg_outputs = np.array(case_data['seg_outputs'])
            if len(seg_outputs.shape) == 3:
                aggregated_seg = seg_outputs[0]
            else:
                aggregated_seg = seg_outputs.mean(axis=0)

        # 计算Dice (2.5D模式计算average 2D Dice，3D/case模式计算3D Dice)
        vis_mask = None  # 用于可视化的mask
        if len(case_data['masks']) > 0:
            if args.train_mode == 'slice':
                # 2.5D模式: 计算每个预测slice的2D Dice，然后取平均
                # 聚合后的segmentation只有中间部分有有效预测
                mask_shape = case_data['masks'][0].shape
                if len(mask_shape) == 3:  # (D, H, W)
                    gt_mask_3d = case_data['masks'][0]

                    # 确保尺寸一致
                    if gt_mask_3d.shape != (depth, args.image_size, args.image_size):
                        gt_mask_3d = resize_volume(
                            gt_mask_3d,
                            (depth, args.image_size, args.image_size)
                        )

                    vis_mask = gt_mask_3d  # 保存用于可视化

                    # 对于2.5D模型，只计算有胰腺GT的slice的2D Dice
                    # 关键修复：只计算gt_slice.max() > 0的slice，避免无胰腺slice拉低指标
                    valid_slices = []
                    slice_dice_scores = []

                    for d_idx in range(depth):
                        pred_slice = aggregated_seg[d_idx]
                        gt_slice = gt_mask_3d[d_idx]

                        # 只计算有胰腺GT的slice（关键修复）
                        if gt_slice.max() > 0:
                            dice_2d = compute_2d_dice(pred_slice, gt_slice)
                            slice_dice_scores.append(dice_2d)
                            valid_slices.append(d_idx)

                    # 使用平均2D Dice（只针对有胰腺的slice）
                    if len(slice_dice_scores) > 0:
                        dice = np.mean(slice_dice_scores)
                    else:
                        dice = 0.0
                else:
                    # 退化为2D
                    vis_mask = case_data['masks'][0]
                    dice = compute_2d_dice(aggregated_seg, case_data['masks'][0])
            else:
                # case/3D模式: 计算3D Dice
                if len(case_data['masks'][0].shape) == 4:
                    aggregated_mask = case_data['masks'][0][0]
                else:
                    aggregated_mask = case_data['masks'][0]

                vis_mask = aggregated_mask  # 保存用于可视化

                # 确保mask和预测尺寸一致
                if aggregated_mask.shape != aggregated_seg.shape:
                    aggregated_mask = resize_volume(aggregated_mask, aggregated_seg.shape)
                    vis_mask = aggregated_mask

                # 计算3D Dice
                seg_pred_binary = (aggregated_seg > 0.5).astype(np.float32)
                seg_gt_binary = (aggregated_mask > 0.5).astype(np.float32)
                dice = compute_3d_dice(seg_pred_binary, seg_gt_binary)

            seg_dice_scores.append(dice)

            # 保存可视化数据 (随机选择一个case)
            if case_id == selected_case_for_vis and output_dir is not None and vis_mask is not None:
                vis_case_id = case_id
                vis_data = {
                    'seg_pred': aggregated_seg,
                    'seg_gt': vis_mask,
                    'cls_pred': pred_label,
                    'cls_prob': pred_prob,
                    'cls_gt': case_data['label']
                }
        else:
            # 没有真实mask，使用简化计算
            seg_pred_binary = (aggregated_seg > 0.5).astype(np.float32)
            foreground_ratio = seg_pred_binary.mean()
            seg_dice_scores.append(0.5 + foreground_ratio * 0.5)

    # 保存可视化结果
    if vis_data is not None and output_dir is not None:
        save_visualization(vis_data, vis_case_id, epoch, output_dir)

    # 如果没有case聚合数据，回退到batch级别统计
    if len(all_preds) == 0:
        print("Warning: No case predictions collected!")
        return {
            'loss': total_loss / max(len(dataloader), 1),
            'seg_loss': total_seg_loss / max(len(dataloader), 1),
            'cls_loss': total_cls_loss / max(len(dataloader), 1),
            'cls_acc': 0.0,
            'cls_precision': 0.0,
            'cls_recall': 0.0,
            'cls_f1': 0.0,
            'cls_auc': 0.0,
            'seg_dice': 0.0
        }

    # 分类指标
    cls_acc = accuracy_score(all_labels, all_preds)
    cls_precision = precision_score(all_labels, all_preds, zero_division=0)
    cls_recall = recall_score(all_labels, all_preds, zero_division=0)
    cls_f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        cls_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    except:
        cls_auc = 0.0

    avg_dice = np.mean(seg_dice_scores) if seg_dice_scores else 0.0

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_seg_loss = total_seg_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches

    # 打印分布信息 (指标详情在主循环中统一打印)
    print(f"  [Val Distribution] Pred: Pos={sum(all_preds)}, Neg={len(all_preds)-sum(all_preds)} | "
          f"True: Pos={sum(all_labels)}, Neg={len(all_labels)-sum(all_labels)}")

    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'cls_loss': avg_cls_loss,
        'cls_acc': cls_acc,
        'cls_precision': cls_precision,
        'cls_recall': cls_recall,
        'cls_f1': cls_f1,
        'cls_auc': cls_auc,
        'seg_dice': avg_dice
    }


def compute_dice(pred, target, smooth=1e-5):
    """计算Dice系数"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_3d_dice(pred, target, smooth=1e-5):
    """
    计算3D Dice系数
    Args:
        pred: (D, H, W) 二值化预测
        target: (D, H, W) 二值化真实标签
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return float(dice)


def compute_2d_dice(pred, target, smooth=1e-5):
    """
    计算2D Dice系数
    Args:
        pred: (H, W) 预测概率或二值化预测
        target: (H, W) 二值化真实标签
    """
    # 二值化预测（如果不是的话）
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return float(dice)


def resize_volume(volume, target_shape):
    """
    重采样3D体积到目标形状
    Args:
        volume: (D, H, W) numpy array
        target_shape: (D, H, W) tuple
    """
    import torch.nn.functional as F
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def save_visualization(vis_data, case_id, epoch, output_dir):
    """
    保存可视化结果
    Args:
        vis_data: dict with 'seg_pred', 'seg_gt', 'cls_pred', 'cls_prob', 'cls_gt'
        case_id: case ID
        epoch: epoch number
        output_dir: base output directory
    """
    # 创建可视化文件夹
    vis_dir = Path(output_dir) / f"epoch_{epoch}_val_visual_result"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 保存分割结果为nii文件
    seg_pred = vis_data['seg_pred']  # (D, H, W)
    seg_gt = vis_data['seg_gt']  # (D, H, W)

    # 创建多标签nii: 0=背景, 1=胰腺真实标签(GT), 2=胰腺预测结果(Pred), 3=重叠区域
    combined_label = np.zeros_like(seg_pred, dtype=np.uint8)
    combined_label[seg_gt > 0.5] = 1  # 真实标签
    combined_label[seg_pred > 0.5] = 2  # 预测结果
    overlap = (seg_gt > 0.5) & (seg_pred > 0.5)
    combined_label[overlap] = 3  # 重叠区域

    # 保存为nii
    nii_path = vis_dir / f"case_{case_id:02d}_segmentation.nii.gz"
    sitk_img = sitk.GetImageFromArray(combined_label)
    sitk_img.SetSpacing([1.0, 1.0, 1.0])  # 设置间距
    sitk.WriteImage(sitk_img, str(nii_path))

    # 同时保存预测概率图 (便于查看置信度)
    pred_nii_path = vis_dir / f"case_{case_id:02d}_prediction_prob.nii.gz"
    sitk_pred = sitk.GetImageFromArray(seg_pred.astype(np.float32))
    sitk_pred.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(sitk_pred, str(pred_nii_path))

    # 保存分类结果到csv (每个epoch独立文件，避免混乱)
    csv_path = vis_dir / f"case_{case_id:02d}_classification.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'case_id', 'gt_label', 'pred_label', 'pred_prob_class_1'])
        writer.writerow([
            epoch,
            case_id,
            vis_data['cls_gt'],
            vis_data['cls_pred'],
            f"{vis_data['cls_prob']:.4f}"
        ])

    print(f"  [Vis] Saved visualization for case {case_id:02d} to {vis_dir}")
    print(f"        - Segmentation: {nii_path.name}")
    print(f"        - Classification: {csv_path.name}")


def main(args):
    """主训练函数"""
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: GPU not available, using CPU (training will be slow)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # 数据加载
    train_loader, val_loader = get_dataloaders(args)

    # 创建模型
    print("\nCreating model...")
    # 处理pretrained参数: 可以是布尔值或权重文件路径
    if args.pretrained_weights:
        pretrained = args.pretrained_weights  # 使用自定义权重路径
        print(f"Using custom pretrained weights: {pretrained}")
    else:
        pretrained = args.pretrained  # 使用布尔值控制ImageNet权重
        print(f"Using ImageNet pretrained: {pretrained}")

    if args.model_type == '2.5d':
        model = ResNet25DMultiTask(
            num_seg_classes=1,
            num_cls_classes=2,
            slices_per_input=args.slices_per_input,
            use_attention=args.use_attention,
            pretrained=pretrained,
            dropout=args.dropout,
            use_roi_masking=args.use_roi_masking
        ).to(device)
    else:  # 3d
        model = ResNet253DMultiTask(
            num_seg_classes=1,
            num_cls_classes=2,
            depth=args.depth,
            use_attention=args.use_attention,
            pretrained=pretrained,
            dropout=args.dropout
        ).to(device)

    # 计算类别权重 (基于训练集分布)
    # 注意: 对于slice模式，需要从base dataset获取原始case的label
    if hasattr(train_loader.dataset, 'base_dataset'):
        # Slice2p5DDataset模式
        train_base = train_loader.dataset.base_dataset
        train_labels = [sample['gata6_label'] for sample in train_base.samples]
    elif hasattr(train_loader.dataset, 'samples'):
        # 直接是dataset
        train_labels = [sample['gata6_label'] for sample in train_loader.dataset.samples]
    else:
        # Subset模式 (case模式)
        from torch.utils.data import Subset
        base_dataset = train_loader.dataset.dataset
        train_labels = [base_dataset.samples[i]['gata6_label']
                       for i in train_loader.dataset.indices]

    class_counts = np.bincount(train_labels)
    print(f"\nTraining set class distribution: {class_counts}")

    # 使用逆频率计算权重
    total_samples = len(train_labels)
    n_classes = len(class_counts)
    cls_weights = torch.tensor([
        total_samples / (n_classes * class_counts[i])
        for i in range(n_classes)
    ], dtype=torch.float32).to(device)
    print(f"Class weights: {cls_weights.cpu().tolist()}")

    # 损失函数
    criterion = MultiTaskLoss(
        seg_weight=args.seg_weight,
        cls_weight=args.cls_weight,
        use_dice=True,
        use_focal=args.use_focal,
        cls_weights=cls_weights
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 早停 - 使用F1作为主要指标
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    print(f"Early stopping: patience={args.patience}, metric=F1 score")

    # 训练循环
    print("\nStarting training...")
    best_metric = 0
    start_time = time.time()

    # 使用tqdm包装epoch循环
    epoch_pbar = tqdm(range(args.epochs), desc='Training', position=0)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )

        val_metrics = validate(model, val_loader, criterion, device, args,
                               cls_threshold=args.cls_threshold, epoch=epoch, output_dir=args.output_dir)

        scheduler.step()

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_seg', train_metrics['seg_loss'], epoch)
        writer.add_scalar('Loss/val_seg', val_metrics['seg_loss'], epoch)
        writer.add_scalar('Loss/train_cls', train_metrics['cls_loss'], epoch)
        writer.add_scalar('Loss/val_cls', val_metrics['cls_loss'], epoch)

        writer.add_scalar('Metrics/train_acc', train_metrics['cls_acc'], epoch)
        writer.add_scalar('Metrics/train_f1', train_metrics.get('cls_f1', 0), epoch)
        writer.add_scalar('Metrics/val_acc', val_metrics['cls_acc'], epoch)
        writer.add_scalar('Metrics/val_precision', val_metrics['cls_precision'], epoch)
        writer.add_scalar('Metrics/val_recall', val_metrics['cls_recall'], epoch)
        writer.add_scalar('Metrics/val_f1', val_metrics['cls_f1'], epoch)
        writer.add_scalar('Metrics/val_auc', val_metrics['cls_auc'], epoch)
        writer.add_scalar('Metrics/val_dice', val_metrics['seg_dice'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        epoch_time = time.time() - epoch_start

        # 更新外层进度条信息 (best_metric指的是F1)
        epoch_pbar.set_postfix({
            'best_F1': f"{best_metric:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}",
            'cls_F1': f"{val_metrics['cls_f1']:.4f}",
            'seg_Dice': f"{val_metrics['seg_dice']:.4f}"
        })

        # 打印详细日志 (明确标注分类/分割指标)
        tqdm.write(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        tqdm.write(f"  [Train] Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['cls_acc']:.4f}, F1: {train_metrics.get('cls_f1', 0):.4f}")
        tqdm.write(f"  [Val]   Loss: {val_metrics['loss']:.4f}")
        tqdm.write(f"  [Classification] Acc: {val_metrics['cls_acc']:.4f}, Precision: {val_metrics['cls_precision']:.4f}, Recall: {val_metrics['cls_recall']:.4f}, F1: {val_metrics['cls_f1']:.4f}, AUC: {val_metrics['cls_auc']:.4f}")
        tqdm.write(f"  [Segmentation] Dice: {val_metrics['seg_dice']:.4f}")

        # 保存最佳模型 (使用F1作为主要指标)
        current_metric = val_metrics['cls_f1']
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> Saved best model (F1: {best_metric:.4f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'args': vars(args)
        }, os.path.join(args.output_dir, 'latest_model.pth'))

        if early_stopping(current_metric):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best F1: {best_metric:.4f}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 2.5D/3D Multi-Task Model')

    # 数据参数
    parser.add_argument('--data_root', type=str,
                        default='/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/zs')
    parser.add_argument('--label_file', type=str,
                        default='/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/GATA6_label.xlsx')
    parser.add_argument('--output_dir', type=str, default='./output_v2')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=32, help='Z方向固定层数')
    parser.add_argument('--z_extension', type=int, default=3, help='Z方向扩展层数')
    parser.add_argument('--normalize_mode', type=str, default='window',
                        choices=['window', 'organ', 'global'])
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # 模型参数
    parser.add_argument('--model_type', type=str, default='2.5d',
                        choices=['2.5d', '3d'])
    parser.add_argument('--slices_per_input', type=int, default=3,
                        help='2.5D模式下每输入的slice数')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用ImageNet预训练权重')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='自定义预训练权重文件路径 (如医学影像预训练权重)，设置此项将覆盖--pretrained')
    parser.add_argument('--use_attention', action='store_true', default=True)
    parser.add_argument('--use_roi_masking', action='store_true', default=True,
                        help='分类分支使用ROI Masking，只在胰腺区域内池化特征')
    parser.add_argument('--dropout', type=float, default=0.3)

    # 训练模式
    parser.add_argument('--train_mode', type=str, default='slice',
                        choices=['case', 'slice'],
                        help='case: 每个样本是整个3D volume; slice: 每个样本是2.5D slice group')
    parser.add_argument('--stride', type=int, default=1,
                        help='2.5D模式下滑动窗口步长')

    # 损失权重
    parser.add_argument('--seg_weight', type=float, default=1.0)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--use_focal', action='store_true', default=True,
                        help='使用Focal Loss处理类别不平衡')
    parser.add_argument('--cls_threshold', type=float, default=0.4,
                        help='分类阈值，对于不平衡数据建议0.3-0.4')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)

    args = parser.parse_args()

    main(args)
