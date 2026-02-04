"""
数据分布检查脚本 - 诊断类别不平衡问题
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import Counter

from dataset_v2 import PancreasCaseDataset, Slice2p5DDataset

def check_case_distribution(label_file):
    """检查Case级别的类别分布"""
    df = pd.read_excel(label_file)
    labels = df['GATA6'].values

    class_counts = Counter(labels)
    total = len(labels)

    print("=" * 60)
    print("Case-Level GATA6 Distribution")
    print("=" * 60)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        ratio = count / total * 100
        print(f"  Class {cls}: {count} cases ({ratio:.1f}%)")
    print(f"  Total: {total} cases")

    # 计算类别不平衡比例
    if len(class_counts) == 2:
        majority = max(class_counts.values())
        minority = min(class_counts.values())
        imbalance_ratio = majority / minority
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")

    return class_counts

def check_slice_distribution(data_root, label_file, target_size=(256, 256, 32)):
    """检查Slice级别的类别分布"""
    print("\n" + "=" * 60)
    print("Loading dataset to check slice-level distribution...")
    print("=" * 60)

    base_dataset = PancreasCaseDataset(
        data_root=data_root,
        label_file=label_file,
        target_size=target_size,
        z_extension=3,
        normalize_mode='window'
    )

    # 统计每个case的label
    case_labels = [sample['gata6_label'] for sample in base_dataset.samples]
    case_label_counts = Counter(case_labels)

    print("\nCase-level distribution in loaded dataset:")
    for cls in sorted(case_label_counts.keys()):
        count = case_label_counts[cls]
        print(f"  Class {cls}: {count} cases")

    # 创建2.5D数据集并统计
    slice_dataset = Slice2p5DDataset(base_dataset, slices_per_input=3, stride=1)

    # 统计slice级别的label分布
    slice_labels = []
    for i in range(len(slice_dataset)):
        sample = slice_dataset[i]
        slice_labels.append(sample['gata6_label'].item())

    slice_label_counts = Counter(slice_labels)

    print("\nSlice-level distribution (2.5D dataset):")
    total_slices = len(slice_labels)
    for cls in sorted(slice_label_counts.keys()):
        count = slice_label_counts[cls]
        ratio = count / total_slices * 100
        print(f"  Class {cls}: {count} slices ({ratio:.1f}%)")
    print(f"  Total: {total_slices} slices")

    # 计算每个case产生的slice数量
    target_d = target_size[2]
    slices_per_case = target_d - 3 + 1  # slices_per_input=3, stride=1
    print(f"\n  Each case generates {slices_per_case} slices")
    print(f"  Class imbalance ratio in slices: same as case-level")

    return slice_label_counts

def compute_class_weights(class_counts):
    """计算类别权重 (Inverse Frequency Weighting)"""
    total = sum(class_counts.values())
    n_classes = len(class_counts)

    # 方法1: 逆频率归一化
    weights = {}
    for cls in sorted(class_counts.keys()):
        weights[cls] = total / (n_classes * class_counts[cls])

    print("\n" + "=" * 60)
    print("Suggested Class Weights (Inverse Frequency)")
    print("=" * 60)
    for cls in sorted(weights.keys()):
        print(f"  Class {cls}: {weights[cls]:.4f}")

    # 转换为tensor格式
    weight_tensor = torch.tensor([weights[i] for i in range(n_classes)], dtype=torch.float32)
    print(f"\n  PyTorch weight tensor: {weight_tensor.tolist()}")

    return weight_tensor

def check_roi_mask_distribution(data_root, label_file, target_size=(256, 256, 32)):
    """检查ROI Mask的分布情况"""
    print("\n" + "=" * 60)
    print("Checking ROI Mask Statistics")
    print("=" * 60)

    base_dataset = PancreasCaseDataset(
        data_root=data_root,
        label_file=label_file,
        target_size=target_size,
        z_extension=3,
        normalize_mode='window'
    )

    # 检查几个样本的mask统计
    foreground_ratios = []
    for i in range(min(10, len(base_dataset))):
        sample = base_dataset[i]
        pancreas_mask = sample['pancreas_mask'].numpy()

        # 统计每个slice的前景比例
        for d in range(pancreas_mask.shape[1]):
            slice_mask = pancreas_mask[0, d, :, :]
            fg_ratio = slice_mask.mean()
            foreground_ratios.append(fg_ratio)

    foreground_ratios = np.array(foreground_ratios)
    print(f"\n  Sampled {len(foreground_ratios)} slices from 10 cases")
    print(f"  Mean foreground ratio: {foreground_ratios.mean():.4f}")
    print(f"  Median foreground ratio: {np.median(foreground_ratios):.4f}")
    print(f"  Max foreground ratio: {foreground_ratios.max():.4f}")
    print(f"  Min foreground ratio: {foreground_ratios.min():.4f}")
    print(f"  Slices with zero foreground: {(foreground_ratios == 0).sum()} ({(foreground_ratios == 0).mean()*100:.1f}%)")

def main():
    data_root = '/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/zs'
    label_file = '/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6_muti_tasks/data/GATA6_label.xlsx'

    # 1. 检查case级别分布
    class_counts = check_case_distribution(label_file)

    # 2. 检查slice级别分布
    slice_counts = check_slice_distribution(data_root, label_file)

    # 3. 计算建议的类别权重
    weights = compute_class_weights(class_counts)

    # 4. 检查ROI mask分布
    check_roi_mask_distribution(data_root, label_file)

    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)

    if len(class_counts) == 2:
        ratio = max(class_counts.values()) / min(class_counts.values())
        if ratio > 2:
            print(f"⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print(f"   Ratio: {ratio:.1f}:1")
            print(f"\n🔧 Recommendations:")
            print(f"   1. Use class weights: {weights.tolist()}")
            print(f"   2. Use Focal Loss with alpha=0.25, gamma=2.0")
            print(f"   3. Consider oversampling minority class")
            print(f"   4. Adjust threshold from 0.5 to balance precision/recall")
        else:
            print(f"✓ Class distribution is relatively balanced ({ratio:.1f}:1)")

if __name__ == '__main__':
    main()
