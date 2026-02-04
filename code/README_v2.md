# 胰腺分割与GATA6分类 - Case级别2.5D/3D多任务模型

基于ResNet18的Case级别2.5D模型，支持多器官ROI提取和固定尺寸重采样。

## 主要特性

1. **多器官ROI提取**: 使用胰腺+肝脏+脾脏(mask label 1,2,3)共同确定ROI范围
2. **固定层数**: 通过ROI上下扩展3层，然后重采样到固定深度(默认32层)
3. **统一尺寸**: 所有Case重采样到相同尺寸 (256×256×32)
4. **2.5D输入**: 将连续多个slice作为通道输入ResNet18

## 数据预处理流程

```
原始图像 (D×H×W, 各患者不同)
    ↓
多器官mask提取 (label 1,2,3)
    ↓
计算联合bounding box
    ↓
Z方向扩展3层 (z_min-3, z_max+3)
    ↓
裁剪ROI
    ↓
重采样到固定尺寸 (256×256×32)
    ↓
CT窗宽窗位归一化 (WW=400, WL=40)
    ↓
输入模型
```

## 器官Label定义

organ_label.nii中的标签:
- 0: 背景
- 1: **胰腺** (主要分割目标)
- 2: 肝脏
- 3: 脾脏

lesion_mask中的标签:
- 用于病灶位置参考
- 病灶的基因分类标签在GATA6_label.xlsx中

## 文件说明

| 文件 | 功能 |
|------|------|
| `dataset_v2.py` | Case级别数据集，支持多器官ROI和固定尺寸重采样 |
| `model_v2.py` | 2.5D和3D多任务模型 |
| `train_v2.py` | 训练脚本，支持case和slice两种训练模式 |
| `evaluate_v2.py` | 评估脚本 |

## 快速使用

### 1. 训练2.5D模型 (推荐)

使用 `CaseLevel2p5DDataset` (depth=32, slices_per_input=4，完美匹配)：

```bash
python train_v2.py \
    --data_root ../data/zs \
    --label_file ../data/GATA6_label.xlsx \
    --output_dir ../output_25d \
    --model_type 2.5d \
    --train_mode slice \
    --slices_per_input 4 \
    --image_size 256 \
    --depth 32 \
    --z_extension 3 \
    --pretrained \
    --batch_size 16 \
    --epochs 100
```

- `--pretrained`: 使用默认 ImageNet1K-V1 预训练权重
- 不使用 `--pretrained`: 从零开始训练
- `--pretrained_weights /path/to/weights.pth`: 使用自定义权重

或使用滑动窗口版本 (depth=32, slices_per_input=3，需要聚合)：

```bash
python train_v2.py \
    --data_root ../data/zs \
    --label_file ../data/GATA6_label.xlsx \
    --output_dir ../output_25d \
    --model_type 2.5d \
    --train_mode slice \
    --slices_per_input 3 \
    --image_size 256 \
    --depth 32 \
    --z_extension 3 \
    --pretrained \
    --batch_size 16 \
    --epochs 100
```

### 2. 训练3D模型

```bash
python train_v2.py \
    --data_root ../data/zs \
    --label_file ../data/GATA6_label.xlsx \
    --output_dir ../output_3d \
    --model_type 3d \
    --train_mode case \
    --image_size 256 \
    --depth 32 \
    --z_extension 3 \
    --batch_size 4 \
    --epochs 100
```

### 3. 评估模型

```bash
python evaluate_v2.py \
    --mode dataset \
    --checkpoint ../output_25d/best_model.pth \
    --output_dir ../results
```

### 4. 单患者推理

```bash
python evaluate_v2.py \
    --mode patient \
    --checkpoint ../output_25d/best_model.pth \
    --patient_dir ../data/zs/01芮秋芬 \
    --output_dir ../results
```

## 关键参数说明

### 数据参数
- `--target_size`: 重采样后的固定尺寸 (默认256×256×32)
- `--z_extension`: Z方向扩展层数 (默认3层)
- `--normalize_mode`: 归一化方式 ('window', 'organ', 'global')

### 模型参数
- `--model_type`: '2.5d' 或 '3d'
- `--slices_per_input`: 2.5D模式下每个输入包含的slice数 (默认3)
- `--train_mode`: 'case' (整个3D) 或 'slice' (2.5D切片组)

### 损失权重
- `--seg_weight`: 分割任务权重
- `--cls_weight`: 分类任务权重

## 关于 `depth` 和 `slices_per_input` 的关系

### 问题
当 `depth=32` 而 `slices_per_input=3` 时，32不能被3整除，会产生30个重叠样本需要聚合。

### 推荐方案: 使用 `CaseLevel2p5DDataset`

最简单的方法是**调整参数使它们匹配**，每个case生成不重叠的2.5D组：

```python
# depth=32, slices_per_input=4 -> 正好8个不重叠的组
dataset = CaseLevel2p5DDataset(base_dataset, slices_per_input=4)
```

### 32的因数选择

| slices_per_input | groups | 上下文范围 |
|-----------------|--------|-----------|
| 1 | 32 | 单层 (2D) |
| 2 | 16 | 2层 |
| **4** | **8** | **4层 (推荐)** |
| 8 | 4 | 8层 |
| 16 | 2 | 16层 |
| 32 | 1 | 整个体积 (3D) |

### 备选方案: 滑动窗口聚合

如果必须用3层上下文，使用 `Slice2p5DDataset` + `PredictionsAggregator`：

```python
# 训练 - 生成30个样本
dataset = Slice2p5DDataset(base_dataset, slices_per_input=3, stride=1)

# 推理 - 聚合重叠预测
aggregator = PredictionsAggregator(depth=32, slices_per_input=3, stride=1)
aggregated_cls = aggregator.aggregate_classification(all_cls_probs)
aggregated_seg = aggregator.aggregate_segmentation(seg_outputs, slice_indices, (32, 256, 256))
```

### 推荐配置

| depth | slices_per_input | 数据集类 | 说明 |
|-------|-----------------|----------|------|
| 32 | 4 | CaseLevel2p5DDataset | **推荐**，4层上下文，8个样本 |
| 32 | 3 | Slice2p5DDataset | 30个样本，需要聚合 |
| 32 | 1 | Slice2p5DDataset | 纯2D，32个样本 |

## 两种训练模式对比

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **case** | 每个样本是整个3D volume | 3D模型，GPU内存充足 |
| **slice** | 每个样本是2.5D slice group | 数据增强更多，batch size更大 |

## 模型架构

```
输入 (B, slices_per_input, H, W)
    ↓
ResNet18编码器 (共享)
    ↓
┌──────────────┬──────────────┐
↓              ↓              ↓
分割分支      主分类器       辅助分类器
(解码器)      (注意力+GAP)   (分割特征)
    ↓              ↓              ↓
胰腺Mask     分类结果1     分类结果2
(1, H, W)     ↓              ↓
         融合分类输出 (加权平均)
```

## 预训练权重处理

### ImageNet预训练 (默认)

使用 `--pretrained` 启用 ImageNet1K-V1 预训练权重：

```bash
python train_v2.py --pretrained ...
```

对于2.5D模型，输入通道数=slices_per_input:
- 将ImageNet预训练权重的3通道平均
- 复制到slices_per_input通道
- 进行数值缩放保持范围一致

### 不使用预训练

从零开始训练：

```bash
python train_v2.py --no-pretrained ...
```

### 自定义医学影像预训练权重

**注意**: 当前2.5D/3D模型使用2D ResNet18架构，需要2D预训练权重。
3D MedicalNet权重（使用3D卷积核）不兼容。

如需加载自定义2D预训练权重：

```bash
python train_v2.py \
    --pretrained_weights /path/to/custom_resnet18.pth \
    ...
```

自定义权重文件支持以下格式：
- 直接包含 `state_dict` 的字典
- 包含 `state_dict` / `model` / `model_state_dict` 键的checkpoint
- 包含 `module.` 前缀的权重（会自动移除）

## 训练进度显示

训练脚本使用 `tqdm` 显示进度条：

```
Training:  10%|██▎               | 10/100 [05:23<48:21, best_metric: 0.8234]
[Train]  50%|███████▌          | 100/200 [02:15<02:15, loss: 0.5234, seg: 0.3123, cls: 0.2111]
```

- 外层进度条: 显示总体 epoch 进度
- 内层进度条: 显示每个 epoch 中的 batch 进度
- 实时显示损失值、评估指标

## 注意事项

1. **内存优化**: 如果GPU内存不足，可以减小`--depth`或`--batch_size`
2. **类别不平衡**: 启用`--use_weighted_sampler`处理GATA6标签不平衡
3. **数据增强**: 训练时自动应用随机翻转、旋转、平移
4. **重采样**: 使用trilinear插值对图像，nearest对mask
