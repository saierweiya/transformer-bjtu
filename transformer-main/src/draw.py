
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob


def plot_size_comparison(size_files, labels):
    """绘制大小模型对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_style("whitegrid")

    # 创建子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 颜色配置
    colors = ['#1f77b4', '#d62728']  # 蓝色和红色

    # 线型配置
    line_styles = {
        'train': '-',
        'val': '--'
    }

    for i, (file_path, label) in enumerate(zip(size_files, labels)):
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        df = pd.read_csv(file_path)

        # 调整epoch编号（如果需要）
        if 'epoch' in df.columns and df['epoch'].iloc[0] == 0:
            df['epoch'] += 1

        # 标记点配置
        mark_every = max(1, len(df) // 5) if len(df) > 0 else 1

        # 上图：训练损失
        if 'train_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'],
                     label=f'{label} - Train',
                     color=colors[i],
                     linestyle=line_styles['train'],
                     linewidth=2.5,
                     marker='o' if i == 0 else 's',
                     markevery=mark_every,
                     markersize=6)

        # 下图：验证损失
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'],
                     label=f'{label} - Validation',
                     color=colors[i],
                     linestyle=line_styles['val'],
                     linewidth=2.5,
                     marker='o' if i == 0 else 's',
                     markevery=mark_every,
                     markersize=6,
                     alpha=0.9)

    # 设置图表标题和标签
    ax1.set_title('Model Size Comparison: Large vs Small Transformer',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    ax2.set_title('Validation Loss Comparison: Large vs Small Transformer',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存图片
    output_path = './results/size_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"大小模型对比图已保存至: {output_path}")
    return output_path


def plot_ablation_comparison(ablation_files, labels):
    """绘制消融实验对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_style("whitegrid")

    # 创建子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 颜色配置 - 使用更多颜色
    colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2']  # 绿色, 紫色, 棕色, 粉色

    # 线型配置
    line_styles = {
        'train': '-',
        'val': '--'
    }

    for i, (file_path, label) in enumerate(zip(ablation_files, labels)):
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        df = pd.read_csv(file_path)

        # 调整epoch编号（如果需要）
        if 'epoch' in df.columns and df['epoch'].iloc[0] == 0:
            df['epoch'] += 1

        # 标记点配置
        mark_every = max(1, len(df) // 5) if len(df) > 0 else 1

        # 标记符号
        markers = ['o', 's', '^', 'D']  # 圆形, 方形, 三角形, 菱形

        # 上图：训练损失
        if 'train_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'],
                     label=f'{label} - Train',
                     color=colors[i % len(colors)],
                     linestyle=line_styles['train'],
                     linewidth=2.5,
                     marker=markers[i % len(markers)],
                     markevery=mark_every,
                     markersize=6)

        # 下图：验证损失
        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'],
                     label=f'{label} - Validation',
                     color=colors[i % len(colors)],
                     linestyle=line_styles['val'],
                     linewidth=2.5,
                     marker=markers[i % len(markers)],
                     markevery=mark_every,
                     markersize=6,
                     alpha=0.9)

    # 设置图表标题和标签
    ax1.set_title('Ablation Studies: Training Loss Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    ax2.set_title('Ablation Studies: Validation Loss Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存图片
    output_path = './results/ablation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"消融实验对比图已保存至: {output_path}")
    return output_path


def create_summary_plot_all(files, labels, group_names):
    """创建所有模型的汇总对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 颜色配置
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

    # 标记符号
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 绘制训练损失
    for i, (file_path, label) in enumerate(zip(files, labels)):
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if 'epoch' in df.columns and df['epoch'].iloc[0] == 0:
            df['epoch'] += 1

        mark_every = max(1, len(df) // 5) if len(df) > 0 else 1

        if 'train_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'],
                     label=label,
                     color=colors[i],
                     marker=markers[i % len(markers)],
                     markevery=mark_every,
                     linewidth=2,
                     markersize=5)

    ax1.set_title('Training Loss: All Models', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # 绘制验证损失
    for i, (file_path, label) in enumerate(zip(files, labels)):
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if 'epoch' in df.columns and df['epoch'].iloc[0] == 0:
            df['epoch'] += 1

        mark_every = max(1, len(df) // 5) if len(df) > 0 else 1

        if 'val_loss' in df.columns:
            ax2.plot(df['epoch'], df['val_loss'],
                     label=label,
                     color=colors[i],
                     marker=markers[i % len(markers)],
                     markevery=mark_every,
                     linewidth=2,
                     markersize=5)

    ax2.set_title('Validation Loss: All Models', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = './results/all_models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"所有模型汇总对比图已保存至: {output_path}")
    return output_path


def create_performance_table(files, labels):
    """创建性能对比表格"""
    metrics_data = []

    for file_path, label in zip(files, labels):
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            final_train = df['train_loss'].iloc[-1]
            final_val = df['val_loss'].iloc[-1]
            min_val = df['val_loss'].min()
            converge_epoch = df['val_loss'].idxmin() + 1

            metrics_data.append({
                'Model': label,
                'Final Train Loss': round(final_train, 4),
                'Final Val Loss': round(final_val, 4),
                'Min Val Loss': round(min_val, 4),
                'Epochs to Converge': converge_epoch
            })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = './results/all_models_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)

        print(f"性能对比表格已保存至: {metrics_path}")
        print("\n性能对比:")
        print(metrics_df.to_string(index=False))

        return metrics_df
    else:
        print("没有找到有效的性能数据")
        return None


def find_csv_files(base_dir="./results"):
    """自动查找CSV文件"""
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

    # 过滤出包含训练历史的文件
    training_files = []
    for file in csv_files:
        if any(keyword in os.path.basename(file).lower() for keyword in
               ['train', 'history', 'loss', 'metrics']):
            training_files.append(file)

    return training_files


def main():
    # 确保结果目录存在
    os.makedirs('./results', exist_ok=True)

    # 定义六个CSV文件路径
    file_paths = [
        "../results/standard_Transformer/transformer_large_training_history.csv",
        "../results/Small_Transformer/transformer_small_full_full_training_history.csv",
        "../results/no_residual_Small_Transformer/transformer_small_full_no_residual_training_history.csv",
        "../results/no_layer_norm_Small_Transformer/transformer_small_full_no_layer_norm_training_history.csv",
        "../results/no_positional_encoding_Small_Transformer/transformer_small_full_no_positional_encoding_training_history.csv",
        "../results/single_head_Small_Transformer/transformer_small_full_full_training_history.csv"
    ]

    # 对应的标签
    labels = [
        "Large Model",
        "Small Model",
        "No Residual",
        "No Layer Norm",
        "No Positional Encoding",
        "Single Head"
    ]

    # 检查文件是否存在，如果不存在则尝试自动查找
    missing_files = [path for path in file_paths if not os.path.exists(path)]
    if missing_files:
        print("以下文件不存在，尝试自动查找...")
        for missing in missing_files:
            print(f"  - {missing}")

        # 自动查找CSV文件
        found_files = find_csv_files()
        if found_files:
            print(f"找到 {len(found_files)} 个CSV文件:")
            for f in found_files:
                print(f"  - {f}")

            # 如果找到的文件数量足够，使用它们
            if len(found_files) >= 6:
                file_paths = found_files[:6]
                # 生成对应的标签
                labels = [f"Model {i + 1}" for i in range(6)]
            else:
                print("找到的文件数量不足6个，请检查文件路径")
                return

    # 分组：前两个为大小模型，后四个为消融实验
    size_files = file_paths[:2]
    size_labels = labels[:2]

    ablation_files = file_paths[2:6]
    ablation_labels = labels[2:6]

    # 绘制大小模型对比
    plot_size_comparison(size_files, size_labels)

    # 绘制消融实验对比
    plot_ablation_comparison(ablation_files, ablation_labels)

    # 创建所有模型汇总图
    create_summary_plot_all(file_paths, labels, ["Size Comparison", "Ablation Studies"])

    # 创建性能表格
    create_performance_table(file_paths, labels)

    print("\n所有图表生成完成！")
    print("生成的文件:")
    print("1. size_comparison.png - 大小模型对比图")
    print("2. ablation_comparison.png - 消融实验对比图")
    print("3. all_models_comparison.png - 所有模型汇总对比图")
    print("4. all_models_metrics.csv - 性能指标表格")


if __name__ == "__main__":
    main()