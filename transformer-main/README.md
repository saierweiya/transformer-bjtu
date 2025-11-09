# 📖 项目简介

本仓库包含一个用于在 IWSLT2017 数据集（英语到德语）上进行机器翻译的 Transformer 模型的实现。
一个从零实现的Transformer模型，包含完整的训练框架、消融实验和可视化工具。

## ✨核心特性

- 🏗️ 完整Transformer实现：编码器-解码器架构、多头注意力、位置编码
- 🔬 消融实验支持：无位置编码、单头注意力、无层归一化、无残差连接
- 📊 可视化工具：训练曲线、性能曲线对比
- ⚡ 高效训练：学习率调度
- 🧪 实验管理：完整的实验记录和结果分析



## 环境安装

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/saierweiya/transformer-bjtu.git
    cd transformer-main
    ```

2.  **创建 Python 环境（推荐）：**
    ```bash
    conda create -n transformer_env python=3.10
    conda activate transformer_env
    ```

3.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```

## 数据准备

修改运行脚本或命令行中的 `--data_path` 参数。

## 🧠 模型架构
标准Transformer配置
组件	配置
模型维度 (d_model)	512
注意力头数 (nhead)	8
编码器层数	6
解码器层数	6
前馈网络维度	2048
Dropout	0.1

##🚀 一键运行所有实验

```bash
bash scripts/run.sh
```

1.  训练完整 Transformer 模型（默认参数：`d_model=512`, `nhead=8`, `num_encoder_layers=6`, `num_decoder_layers=6`, `dim_feedforward=2048`）。
2.  训练小型 Transformer 模型（参数：`d_model=256`, `nhead=4`, `num_encoder_layers=3`, `num_decoder_layers=3`, `dim_feedforward=1024`）。
3.  训练小型 Transformer 模型的消融实验（无残差连接）。
4.  训练小型 Transformer 模型的消融实验（无位置编码）。
5.  训练小型 Transformer 模型的消融实验（无多头注意力）。
6.  训练小型 Transformer 模型的消融实验（无层归一化）。
7.  在 `./src/results/` 目录下生成一个消融实验对比图（`ablation_comparison.png`）。
8.  在 `./src/results/` 目录下生成一个所有模型对比图（`all_models_comparison.png`）。
9.  在 `./src/results/` 目录下生成所有模型数据图（`all_models_metrics.csv`）。
10. 在 `./src/results/` 目录下生成大小模型对比图（`size_comparison.png`）。

## 📁 项目结构

```
├── data  
│   └── de-en 
│       ├── train.tags.de-en.de     
│       ├── train.tags.de-en.en    
│       └── DatasetREADME.md 
├── src/
│   ├── Transformer.py     
│   ├── dataloader.py     
│   ├── trainer.py              
│   ├── main.py                
│   └── draw.py         
├── scripts/
│   └── run.sh                 
├── results/
│   ├── standard_Transformer/transformer_large_training_history.csv     
│   ├── Small_Transformer/transformer_small_full_full_training_history.csv     
│   ├── no_residual_Small_Transformer/transformer_small_full_no_residual_training_history.csv              
│   ├── results/no_layer_norm_Small_Transformer/transformer_small_full_no_layer_norm_training_history.csv      
│   ├── no_positional_encoding_Small_Transformer/transformer_small_full_no_positional_encoding_training_history.csv              
│   └── single_head_Small_Transformer/transformer_small_full_full_training_history.csv                   
├── requirements.txt           
└── README.md                   
```

## 📈 实验结果
| 模型变体          | 验证损失 | 训练时间 | 参数量 |
|------------------|----------|----------|--------|
| Transformer-Large | 2.15     | 45 min   | 65 M   |
| Transformer-Small | 2.89     | 25 min   | 18 M   |
| 无位置编码        | 3.42     | 42 min   | 65 M   |
| 单头注意力        | 2.78     | 40 min   | 65 M   |

