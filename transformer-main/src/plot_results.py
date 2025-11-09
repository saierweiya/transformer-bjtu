# src/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_comparison():
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 使用seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    large_history_file = "./results/transformer_large_training_history.csv"
    small_history_file = "./results/transformer_small_training_history.csv"

    # 检查文件是否存在
    if not os.path.exists(large_history_file):
        print(f"Error: {large_history_file} not found.")
        return
    if not os.path.exists(small_history_file):
        print(f"Error: {small_history_file} not found.")
        return

    # 读取数据
    large_df = pd.read_csv(large_history_file)
    small_df = pd.read_csv(small_history_file)

    # 调整epoch编号（如果需要）
    if large_df['epoch'].iloc[0] == 0:
        large_df['epoch'] += 1
    if small_df['epoch'].iloc[0] == 0:
        small_df['epoch'] += 1

    # 创建子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 颜色配置
    colors = {
        'large_train': '#1f77b4',  # 蓝色
        'large_val': '#aec7e8',  # 浅蓝色
        'small_train': '#d62728',  # 红色
        'small_val': '#ff9896'  # 浅红色
    }

    # 线型配置
    line_styles = {
        'train': '-',
        'val': '--'
    }

    # 标记点配置（每隔几个epoch标记一次）
    mark_every = max(1, len(large_df) // 5)

    # 上图：训练损失对比
    ax1.plot(large_df['epoch'], large_df['train_loss'],
             label='Large Model - Train',
             color=colors['large_train'],
             linestyle=line_styles['train'],
             linewidth=2.5,
             marker='o',
             markevery=mark_every,
             markersize=6)

    ax1.plot(small_df['epoch'], small_df['train_loss'],
             label='Small Model - Train',
             color=colors['small_train'],
             linestyle=line_styles['train'],
             linewidth=2.5,
             marker='s',
             markevery=mark_every,
             markersize=6)

    ax1.set_title('Training Loss Comparison: Large vs Small Transformer',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')  # 浅灰色背景

    # 下图：验证损失对比
    ax2.plot(large_df['epoch'], large_df['val_loss'],
             label='Large Model - Validation',
             color=colors['large_val'],
             linestyle=line_styles['val'],
             linewidth=2.5,
             marker='o',
             markevery=mark_every,
             markersize=6,
             alpha=0.9)

    ax2.plot(small_df['epoch'], small_df['val_loss'],
             label='Small Model - Validation',
             color=colors['small_val'],
             linestyle=line_styles['val'],
             linewidth=2.5,
             marker='s',
             markevery=mark_every,
             markersize=6,
             alpha=0.9)

    ax2.set_title('Validation Loss Comparison: Large vs Small Transformer',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')  # 浅灰色背景

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存图片
    output_path = './results/size_ablation_loss_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"美化后的对比图已保存至: {output_path}")


def create_summary_plot():
    """创建汇总对比图（单图显示所有曲线）"""
    large_history_file = "./results/transformer_large_training_history.csv"
    small_history_file = "./results/transformer_small_training_history.csv"

    if not os.path.exists(large_history_file) or not os.path.exists(small_history_file):
        return

    # 读取数据
    large_df = pd.read_csv(large_history_file)
    small_df = pd.read_csv(small_history_file)

    # 调整epoch编号
    if large_df['epoch'].iloc[0] == 0:
        large_df['epoch'] += 1
    if small_df['epoch'].iloc[0] == 0:
        small_df['epoch'] += 1

    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # 定义颜色和样式
    styles = {
        'large_train': {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5},
        'large_val': {'color': '#2E86AB', 'linestyle': '--', 'marker': 'o', 'linewidth': 2, 'alpha': 0.8},
        'small_train': {'color': '#A23B72', 'linestyle': '-', 'marker': 's', 'linewidth': 2.5},
        'small_val': {'color': '#A23B72', 'linestyle': '--', 'marker': 's', 'linewidth': 2, 'alpha': 0.8}
    }

    mark_every = max(1, len(large_df) // 4)

    # 绘制所有曲线
    ax.plot(large_df['epoch'], large_df['train_loss'],
            label='Large Model - Train', **styles['large_train'], markevery=mark_every)
    ax.plot(large_df['epoch'], large_df['val_loss'],
            label='Large Model - Val', **styles['large_val'], markevery=mark_every)
    ax.plot(small_df['epoch'], small_df['train_loss'],
            label='Small Model - Train', **styles['small_train'], markevery=mark_every)
    ax.plot(small_df['epoch'], small_df['val_loss'],
            label='Small Model - Val', **styles['small_val'], markevery=mark_every)

    # 美化图表
    ax.set_title('Transformer Model Size Ablation Study: Training and Validation Loss',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')

    # 设置图例
    ax.legend(fontsize=11, frameon=True, fancybox=True,
              shadow=True, loc='best', ncol=2)

    # 网格和背景
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')

    # 添加说明文本
    min_epoch = min(len(large_df), len(small_df))
    final_large_val = large_df['val_loss'].iloc[-1]
    final_small_val = small_df['val_loss'].iloc[-1]

    text_str = f'Final Validation Loss:\nLarge Model: {final_large_val:.4f}\nSmall Model: {final_small_val:.4f}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存汇总图
    summary_path = './results/size_ablation_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"汇总对比图已保存至: {summary_path}")


def create_performance_table():
    """创建性能对比表格"""
    large_history_file = "./results/transformer_large_training_history.csv"
    small_history_file = "./results/transformer_small_training_history.csv"

    if not os.path.exists(large_history_file) or not os.path.exists(small_history_file):
        return

    # 读取数据
    large_df = pd.read_csv(large_history_file)
    small_df = pd.read_csv(small_history_file)

    # 计算关键指标
    metrics = {
        'Model': ['Large Transformer', 'Small Transformer'],
        'Final Train Loss': [
            large_df['train_loss'].iloc[-1],
            small_df['train_loss'].iloc[-1]
        ],
        'Final Val Loss': [
            large_df['val_loss'].iloc[-1],
            small_df['val_loss'].iloc[-1]
        ],
        'Min Val Loss': [
            large_df['val_loss'].min(),
            small_df['val_loss'].min()
        ],
        'Epochs to Converge': [
            large_df['val_loss'].idxmin() + 1,
            small_df['val_loss'].idxmin() + 1
        ]
    }

    # 创建DataFrame并保存为CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_path = './results/model_comparison_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)

    print(f"性能对比表格已保存至: {metrics_path}")
    print("\n性能对比:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    # 创建所有图表和表格
    plot_comparison()
    create_summary_plot()
    create_performance_table()

    print("\n所有图表生成完成！")
    print("生成的文件:")
    print("1. size_ablation_loss_comparison.png - 分开展示的训练和验证损失")
    print("2. size_ablation_summary.png - 汇总对比图")
    print("3. model_comparison_metrics.csv - 性能指标表格")