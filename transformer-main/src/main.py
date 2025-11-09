import torch
import argparse
from Transformer import Transformer
from dataloader import get_dataloaders
from trainer import Trainer, count_parameters
import os

def main():

    parser = argparse.ArgumentParser(description='Transformer for IWSLT2017')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to the IWSLT2017 data ')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension for the standard model')
    parser.add_argument('--nhead', type=int, default=8, 
                        help='Number of attention heads for the standard model')
    parser.add_argument('--num_encoder_layers', type=int, default=6, 
                        help='Number of encoder layers for the standard model')
    parser.add_argument('--num_decoder_layers', type=int, default=6, 
                        help='Number of decoder layers for the standard model')
    parser.add_argument('--dim_feedforward', type=int, default=2048, 
                        help='Dimension of feedforward network for the standard model')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples to use for training')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_small_model', action='store_true',
                        help='Use the small Transformer model for ablation study')
    parser.add_argument('--ablation_type', type=str, default='full',
                        help='ablation_experiment')

    args = parser.parse_args() # 解析命令行参数 返回值: 包含所有参数值的命名空间对象


    # 随机种子和设备设置
    torch.manual_seed(args.seed)            # 作用: 设置CPU的随机种子，确保CPU上的随机操作可重现
    torch.cuda.manual_seed_all(args.seed)   # 作用: 设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # 数据加载
    os.makedirs(args.save_path, exist_ok=True)
    print("Loading data...")
    train_loader, val_loader, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv = get_dataloaders(
        args.data_path, 
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length
    )
    print(f"~~原词表的大小 ：   Source vocabulary size: {len(src_vocab)}")
    print(f"~~目标词表的大小：   Target vocabulary size: {len(tgt_vocab)}")
    print(f"~~训练集句子数：    Training samples: {len(train_loader.dataset)}")
    print(f"~~验证集句子数：     samples: {len(val_loader.dataset)}")


    if args.use_small_model:
        print("使用小型的Transformer模型         Using Small Transformer Model.")
        model_name = "transformer_small_full_"+args.ablation_type
    else:
        print("使用大型的Transformer模型         Using Standard (Large) Transformer Model.")
        model_name = "transformer_large"

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        ablation_type=args.ablation_type
    )

    print(f"模型{model_name}的参数量为：Model parameters: {count_parameters(model):,}")

    # 开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        d_model=model.d_model,
        lr=args.lr
    )

    print("开始训练   Starting training...")
    trainer.train(epochs=args.epochs)


    # 保存训练到模型结果
    model_save_path = os.path.join(args.save_path, f'{model_name}_checkpoint.pth')
    curves_save_path = os.path.join(args.save_path, f'{model_name}_training_curves.png')
    history_save_path = os.path.join(args.save_path, f'{model_name}_training_history.csv')
    
    trainer.save_checkpoint(model_save_path)
    trainer.plot_training_curves(curves_save_path)
    
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(trainer.train_losses) + 1),
        'train_loss': trainer.train_losses,
        'val_loss': trainer.val_losses,
    })
    history_df.to_csv(history_save_path, index=False)
    
    print("训练完毕！                Training completed!")
    print(f"结果已经被保存到了目录下    Results saved to {args.save_path}")

if __name__ == "__main__":
    main()