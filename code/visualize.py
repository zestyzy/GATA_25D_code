"""
可视化工具 - 显示分割结果和预测
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头环境
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import SimpleITK as sitk


def load_nii(path):
    """加载nii文件"""
    sitk_image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(sitk_image)
    return array


def create_overlay(image, mask, alpha=0.3, color='red'):
    """创建叠加图"""
    plt.figure(figsize=(10, 10))

    # 显示原图
    plt.imshow(image, cmap='gray')

    # 创建彩色mask
    if color == 'red':
        cmap = matplotlib.colors.ListedColormap(['none', 'red'])
    elif color == 'green':
        cmap = matplotlib.colors.ListedColormap(['none', 'green'])
    elif color == 'blue':
        cmap = matplotlib.colors.ListedColormap(['none', 'blue'])
    else:
        cmap = matplotlib.colors.ListedColormap(['none', color])

    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis('off')

    return plt.gcf()


def visualize_patient(image_path, organ_mask_path, lesion_mask_path=None,
                      pred_seg_path=None, output_dir='./visualizations',
                      num_slices=5, case_id=None):
    """
    可视化患者数据

    Args:
        image_path: CT图像路径
        organ_mask_path: 胰腺mask路径
        lesion_mask_path: 病灶mask路径 (可选)
        pred_seg_path: 预测分割路径 (可选)
        output_dir: 输出目录
        num_slices: 显示的切片数量
        case_id: 患者ID
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    image = load_nii(image_path)
    organ_mask = load_nii(organ_mask_path)

    if lesion_mask_path:
        lesion_mask = load_nii(lesion_mask_path)
    else:
        lesion_mask = None

    if pred_seg_path:
        pred_seg = load_nii(pred_seg_path)
    else:
        pred_seg = None

    # 找到包含胰腺的切片
    organ_slices = np.where(np.any(organ_mask, axis=(1, 2)))[0]

    if len(organ_slices) == 0:
        print(f"No pancreas found in {case_id}")
        return

    # 选择要显示的切片
    if len(organ_slices) <= num_slices:
        selected_slices = organ_slices
    else:
        step = len(organ_slices) // num_slices
        selected_slices = organ_slices[::step][:num_slices]

    # 创建可视化
    n_cols = 3 if pred_seg else 2
    if lesion_mask:
        n_cols += 1

    fig, axes = plt.subplots(len(selected_slices), n_cols,
                             figsize=(4*n_cols, 4*len(selected_slices)))

    if len(selected_slices) == 1:
        axes = axes.reshape(1, -1)

    for idx, z in enumerate(selected_slices):
        # 原始图像
        axes[idx, 0].imshow(image[z], cmap='gray')
        axes[idx, 0].set_title(f'Slice {z} - CT')
        axes[idx, 0].axis('off')

        # 胰腺GT
        axes[idx, 1].imshow(image[z], cmap='gray')
        organ_overlay = np.ma.masked_where(organ_mask[z] == 0, organ_mask[z])
        axes[idx, 1].imshow(organ_overlay, cmap='Reds', alpha=0.5)
        axes[idx, 1].set_title('Pancreas GT')
        axes[idx, 1].axis('off')

        col = 2

        # 病灶GT (如果有)
        if lesion_mask:
            axes[idx, col].imshow(image[z], cmap='gray')
            lesion_overlay = np.ma.masked_where(lesion_mask[z] == 0, lesion_mask[z])
            axes[idx, col].imshow(lesion_overlay, cmap='Blues', alpha=0.5)
            axes[idx, 2].set_title('Lesion GT')
            axes[idx, col].axis('off')
            col += 1

        # 预测分割 (如果有)
        if pred_seg is not None:
            axes[idx, col].imshow(image[z], cmap='gray')
            pred_overlay = np.ma.masked_where(pred_seg[z] < 0.5, pred_seg[z])
            axes[idx, col].imshow(pred_overlay, cmap='Greens', alpha=0.5)
            axes[idx, col].set_title('Predicted Seg')
            axes[idx, col].axis('off')

    title = f"Patient {case_id}" if case_id else "Visualization"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # 保存
    output_file = os.path.join(output_dir, f'{case_id or "patient"}_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_file}")


def plot_training_curves(log_file, output_path='./training_curves.png'):
    """绘制训练曲线 (需要TensorBoard日志)"""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_file)
        ea.Reload()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 损失曲线
        if 'Loss/train' in ea.Tags()['scalars']:
            train_loss = ea.Scalars('Loss/train')
            val_loss = ea.Scalars('Loss/val')

            axes[0, 0].plot([x.step for x in train_loss], [x.value for x in train_loss], label='Train')
            axes[0, 0].plot([x.step for x in val_loss], [x.value for x in val_loss], label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()

        # 分类准确率
        if 'Metrics/val_acc' in ea.Tags()['scalars']:
            val_acc = ea.Scalars('Metrics/val_acc')
            axes[0, 1].plot([x.step for x in val_acc], [x.value for x in val_acc])
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')

        # F1和AUC
        if 'Metrics/val_f1' in ea.Tags()['scalars']:
            val_f1 = ea.Scalars('Metrics/val_f1')
            val_auc = ea.Scalars('Metrics/val_auc')

            axes[0, 2].plot([x.step for x in val_f1], [x.value for x in val_f1], label='F1')
            axes[0, 2].plot([x.step for x in val_auc], [x.value for x in val_auc], label='AUC')
            axes[0, 2].set_title('Classification Metrics')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].legend()

        # Dice系数
        if 'Metrics/val_dice' in ea.Tags()['scalars']:
            val_dice = ea.Scalars('Metrics/val_dice')
            axes[1, 0].plot([x.step for x in val_dice], [x.value for x in val_dice])
            axes[1, 0].set_title('Validation Dice')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice')

        # 分割损失
        if 'Loss/val_seg' in ea.Tags()['scalars']:
            seg_loss = ea.Scalars('Loss/val_seg')
            cls_loss = ea.Scalars('Loss/val_cls')

            axes[1, 1].plot([x.step for x in seg_loss], [x.value for x in seg_loss], label='Seg')
            axes[1, 1].plot([x.step for x in cls_loss], [x.value for x in cls_loss], label='Cls')
            axes[1, 1].set_title('Validation Losses')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()

        # 学习率
        if 'LR' in ea.Tags()['scalars']:
            lr = ea.Scalars('LR')
            axes[1, 2].plot([x.step for x in lr], [x.value for x in lr])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('LR')
            axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {output_path}")

    except Exception as e:
        print(f"Error plotting training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--mode', type=str, choices=['patient', 'training'],
                        default='patient')

    # 患者可视化参数
    parser.add_argument('--image', type=str, help='Path to image.nii')
    parser.add_argument('--organ_mask', type=str, help='Path to organ_label.nii')
    parser.add_argument('--lesion_mask', type=str, help='Path to label.nii (optional)')
    parser.add_argument('--pred_seg', type=str, help='Path to predicted segmentation (optional)')
    parser.add_argument('--case_id', type=str, help='Case ID')

    # 训练曲线参数
    parser.add_argument('--log_dir', type=str, help='TensorBoard log directory')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--num_slices', type=int, default=5)

    args = parser.parse_args()

    if args.mode == 'patient':
        if not args.image or not args.organ_mask:
            print("Error: --image and --organ_mask are required for patient mode")
            return

        visualize_patient(
            args.image,
            args.organ_mask,
            args.lesion_mask,
            args.pred_seg,
            args.output_dir,
            args.num_slices,
            args.case_id
        )

    elif args.mode == 'training':
        if not args.log_dir:
            print("Error: --log_dir is required for training mode")
            return

        # 找到最新的日志文件
        import glob
        log_files = glob.glob(os.path.join(args.log_dir, 'events.out.tfevents.*'))
        if not log_files:
            print(f"No TensorBoard logs found in {args.log_dir}")
            return

        latest_log = max(log_files, key=os.path.getctime)
        output_path = os.path.join(args.output_dir, 'training_curves.png')
        os.makedirs(args.output_dir, exist_ok=True)

        plot_training_curves(latest_log, output_path)


if __name__ == '__main__':
    main()
