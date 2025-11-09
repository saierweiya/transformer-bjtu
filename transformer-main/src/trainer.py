import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from Transformer import generate_square_subsequent_mask

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab, 
                 device, d_model=512, lr=0.0003, warmup_steps=2000):
        self.model = model.to(device)           # 模型移到指定设备
        self.train_loader = train_loader        # 训练数据加载器
        self.val_loader = val_loader            # 验证数据加载器
        self.src_vocab = src_vocab              # 源语言词汇表
        self.tgt_vocab = tgt_vocab              # 目标语言词汇表
        self.device = device                    # 计算设备
        self.d_model = d_model                  # 模型维度
        self.warmup_steps = warmup_steps        # 学习率预热步数

        # AdamW优化器，适合Transformer
        # beta1=0.9:  一阶矩估计的指数衰减率  控制梯度均值的衰减速度  影响动量(momentum)的大小
        # beta2=0.98: 二阶矩估计的指数衰减率  控制梯度平方的衰减速度  影响自适应学习率的调整
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        # 自定义学习率调度器（带预热）
        self.scheduler = LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: self._lr_scale(step)
        )
        
        # 损失函数：交叉熵，忽略填充符
        # 总损失 = Σ(loss_i)  对所有有效位置i 的损失
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['<pad>'], reduction='sum')
        self.train_losses = []  # 记录训练损失
        self.val_losses = []    # 记录验证损失
    
    def _lr_scale(self, step):
        """Learning rate scaling function with warmup."""
        step = max(step, 0)
        step += 1               # step最少是1
        return min(step ** -0.5, step * (self.warmup_steps ** -1.5))


    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc="Training")):
            src, tgt = src.to(self.device), tgt.to(self.device)
            """
            原始目标序列: [<sos>, I, love, ML, <eos>]
            tgt_input: [<sos>, I, love, ML]      # 模型输入
            tgt_output: [I, love, ML, <eos>]     # 期望输出
            """
            tgt_input = tgt[:, :-1]  # 目标序列输入（去掉最后一个token）
            tgt_output = tgt[:, 1:]  # 目标序列输出（去掉第一个token）

            # 目标序列掩码（防止看到未来信息） :tgt_mask: 确保解码器只能看到当前位置之前的信息
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            # 源序列掩码（忽略填充位置）      ：src_mask: 在编码器中忽略填充符的计算
            src_mask = (src != self.src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(self.device)
            # 4. 前向传播
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            # 5. 损失计算
            loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            # 6. 统计非填充token数量（用于归一化）
            """
            tgt_output:
            [[23, 45,  2,  0,  0,  0],
             [67,  2,  0,  0,  0,  0],
             [89, 12, 34,  2,  0,  0]]
            
            non_pad_mask:
            [[True,  True,  True, False, False, False],
             [True,  True, False, False, False, False],
             [True,  True,  True,  True, False, False]]
             
            # 1. 计算原始损失（所有位置的交叉熵求和）
            loss = criterion(output, tgt_output)  # 假设得到 loss = 15.2
            # 2. 统计有效token数量
            num_tokens = 9  # 从上面的例子
            # 3. 归一化损失
            normalized_loss = loss / num_tokens  # 15.2 / 9 = 1.6889
            """
            non_pad_mask = (tgt_output != self.tgt_vocab['<pad>'])
            num_tokens = non_pad_mask.sum().item()
            # 7. 数值稳定性检查
            if torch.isnan(loss):
                print(f"NaN found in loss at batch {batch_idx}!")
                print("Output sample:", output.view(-1, output.size(-1))[:10, :10]) 
                print("Target sample:", tgt_output.contiguous().view(-1)[:10]) 
                continue
            # 8. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 9. 梯度裁剪（防止梯度爆炸）
            # 计算所有参数梯度的L2范数 # 只有范数超过max_norm时才裁剪 # 按比例缩放梯度
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"NaN/Inf gradient norm found at batch {batch_idx}!")
                continue
            # 10. 参数更新和学习率调整
            self.optimizer.step()
            self.scheduler.step()
            # 11. 损失累计
            total_loss += loss.item()
            total_tokens += num_tokens

        # 12. 计算平均损失
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = float('inf') 
        
        self.train_losses.append(avg_loss)
        return avg_loss

    
    def validate(self):
        self.model.eval() # 切换到评估模式
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad(): # 禁用梯度计算
            for src, tgt in tqdm(self.val_loader, desc="Validating"):
                # 与训练相同的前向过程，但不更新参数
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
                src_mask = (src != self.src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(self.device)
                
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                
                if torch.isnan(loss):
                    print("NaN found in validation loss!")
                    continue
                
                non_pad_mask = (tgt_output != self.tgt_vocab['<pad>'])
                num_tokens = non_pad_mask.sum().item()

                total_loss += loss.item()
                total_tokens += num_tokens

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            avg_loss = float('inf')
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()  # 训练一个epoch
            val_loss = self.validate()  # 验证
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)


    """
        模型保存和加载
    """
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def plot_training_curves(self, save_path):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        ax.plot(self.train_losses, label='Train Loss', color='blue')
        ax.plot(self.val_losses, label='Validation Loss', color='red')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    # tensor.numel() 计算张量中元素的总数 布尔标志，指示参数是否需要在训练中更新
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


