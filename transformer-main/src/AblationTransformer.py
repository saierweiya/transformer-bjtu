import torch
import torch.nn as nn
import math
import copy
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AblationTransformer(nn.Module):
    """
    支持消融实验的Transformer变体
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=512,
                 ablation_type="full",  # full, no_pos_encoding, single_head, no_layer_norm, no_residual
                 single_head_layer_idx=None):  # 指定哪些层使用单头注意力
        super(AblationTransformer, self).__init__()

        self.d_model = d_model
        self.ablation_type = ablation_type

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

        # 位置编码消融
        if ablation_type != "no_pos_encoding":
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        else:
            self.pos_encoder = None

        # 编码器层
        encoder_layers = []
        for i in range(num_encoder_layers):
            # 单头注意力消融
            if ablation_type == "single_head" and (single_head_layer_idx is None or i in single_head_layer_idx):
                layer_nhead = 1
            else:
                layer_nhead = nhead

            encoder_layer = AblationEncoderLayer(
                d_model, layer_nhead, dim_feedforward, dropout, ablation_type
            )
            encoder_layers.append(encoder_layer)

        self.encoder = AblationEncoder(encoder_layers)

        # 解码器层
        decoder_layers = []
        for i in range(num_decoder_layers):
            # 单头注意力消融
            if ablation_type == "single_head" and (single_head_layer_idx is None or i in single_head_layer_idx):
                layer_nhead = 1
            else:
                layer_nhead = nhead

            decoder_layer = AblationDecoderLayer(
                d_model, layer_nhead, dim_feedforward, dropout, ablation_type
            )
            decoder_layers.append(decoder_layer)

        self.decoder = AblationDecoder(decoder_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # 应用位置编码（除非消融）
        if self.pos_encoder is not None:
            src_emb = self.pos_encoder(src_emb)
            tgt_emb = self.pos_encoder(tgt_emb)

        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        output = self.generator(output)

        return output


class AblationEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, ablation_type="full"):
        super(AblationEncoderLayer, self).__init__()
        self.ablation_type = ablation_type

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化消融
        if ablation_type != "no_layer_norm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, src_mask)

        # 残差连接消融
        if self.ablation_type != "no_residual":
            src = src + self.dropout1(attn_output)
        else:
            src = self.dropout1(attn_output)

        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))

        # 残差连接消融
        if self.ablation_type != "no_residual":
            src = src + self.dropout2(ff_output)
        else:
            src = self.dropout2(ff_output)

        src = self.norm2(src)
        return src


class AblationDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, ablation_type="full"):
        super(AblationDecoderLayer, self).__init__()
        self.ablation_type = ablation_type

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化消融
        if ablation_type != "no_layer_norm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)

        # 残差连接消融
        if self.ablation_type != "no_residual":
            tgt = tgt + self.dropout1(attn_output)
        else:
            tgt = self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.multihead_attn(tgt, memory, memory, memory_mask)

        # 残差连接消融
        if self.ablation_type != "no_residual":
            tgt = tgt + self.dropout2(attn_output)
        else:
            tgt = self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))

        # 残差连接消融
        if self.ablation_type != "no_residual":
            tgt = tgt + self.dropout3(ff_output)
        else:
            tgt = self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt


class AblationEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super(AblationEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class AblationDecoder(nn.Module):
    def __init__(self, decoder_layers):
        super(AblationDecoder, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output


# 保留您原有的PositionalEncoding和MultiHeadAttention类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask_expanded = mask.expand_as(scores)
            scores.masked_fill_(mask_expanded == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        output = self.out_linear(output)

        return output


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# 消融实验训练和评估函数
class AblationExperiment:
    def __init__(self, vocab_size_src, vocab_size_tgt, device):
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.device = device
        self.results = {}

    def train_model(self, model, train_loader, val_loader, epochs=10, model_name="full"):
        print(f"Training {model_name} model...")

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_train_loss = 0
            for batch_idx, (src, tgt) in enumerate(train_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)

                optimizer.zero_grad()

                # 创建目标输入和输出
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 创建掩码
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f'{model_name} - Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

                    output = model(src, tgt_input, tgt_mask=tgt_mask)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f'{model_name} - Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 计算困惑度
        perplexity = math.exp(avg_val_loss)

        self.results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': avg_val_loss,
            'perplexity': perplexity,
            'model': model
        }

        return train_losses, val_losses, perplexity

    def run_ablation_study(self, train_loader, val_loader, epochs=10):
        """运行所有消融实验"""
        ablation_configs = [
            ("full", "完整模型"),
            ("no_pos_encoding", "无位置编码"),
            ("single_head", "单头注意力"),
            ("no_layer_norm", "无层归一化"),
            ("no_residual", "无残差连接")
        ]

        for config, description in ablation_configs:
            print(f"\n=== 开始 {description} 实验 ===")

            # 创建模型
            if config == "single_head":
                # 单头注意力：所有层都使用单头
                model = AblationTransformer(
                    self.vocab_size_src, self.vocab_size_tgt,
                    d_model=512, nhead=8,  # nhead=8但实际使用时会设为1
                    num_encoder_layers=2, num_decoder_layers=2,  # 减少层数以加快训练
                    ablation_type=config,
                    single_head_layer_idx=list(range(6))  # 所有层都用单头
                )
            else:
                model = AblationTransformer(
                    self.vocab_size_src, self.vocab_size_tgt,
                    d_model=512, nhead=8,
                    num_encoder_layers=2, num_decoder_layers=2,  # 减少层数以加快训练
                    ablation_type=config
                )

            model = model.to(self.device)

            # 训练模型
            self.train_model(model, train_loader, val_loader, epochs, config)

        return self.results

    def plot_results(self):
        """绘制消融实验结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制训练曲线
        for config, result in self.results.items():
            ax1.plot(result['train_losses'], label=f'{config}')
        ax1.set_title('Training Loss - Ablation Study')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 绘制验证损失和困惑度对比
        configs = list(self.results.keys())
        val_losses = [self.results[config]['final_val_loss'] for config in configs]
        perplexities = [self.results[config]['perplexity'] for config in configs]

        x = range(len(configs))
        ax2.bar(x, val_losses, alpha=0.7, label='Validation Loss')
        ax2.set_xlabel('Model Variant')
        ax2.set_ylabel('Loss', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45)

        ax2_twin = ax2.twinx()
        ax2_twin.bar(x, perplexities, alpha=0.7, color='red', label='Perplexity')
        ax2_twin.set_ylabel('Perplexity', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')

        ax2.set_title('Final Validation Loss and Perplexity')

        plt.tight_layout()
        plt.savefig('ablation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_comparison(self):
        """打印消融实验对比结果"""
        print("\n" + "=" * 60)
        print("消融实验对比结果")
        print("=" * 60)
        print(f"{'模型变体':<20} {'验证损失':<12} {'困惑度':<12} {'相对性能下降':<15}")
        print("-" * 60)

        full_val_loss = self.results['full']['final_val_loss']
        full_perplexity = self.results['full']['perplexity']

        for config, result in self.results.items():
            val_loss = result['final_val_loss']
            perplexity = result['perplexity']

            if config == 'full':
                relative_decrease = '-'
            else:
                loss_increase = ((val_loss - full_val_loss) / full_val_loss) * 100
                relative_decrease = f"{loss_increase:.1f}%"

            print(f"{config:<20} {val_loss:<12.4f} {perplexity:<12.2f} {relative_decrease:<15}")

        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 假设的词汇表大小（根据您的数据集调整）
    vocab_size_src = 10000  # 源语言词汇表大小
    vocab_size_tgt = 10000  # 目标语言词汇表大小

    # 创建消融实验实例
    experiment = AblationExperiment(vocab_size_src, vocab_size_tgt, device)


    # 创建模拟数据加载器（用您的实际数据替换）
    # 这里使用随机数据作为示例
    def create_dummy_dataloader(batch_size=32, seq_len=20, num_batches=100):
        src_data = torch.randint(1, vocab_size_src, (num_batches, batch_size, seq_len))
        tgt_data = torch.randint(1, vocab_size_tgt, (num_batches, batch_size, seq_len + 1))
        dataset = list(zip(src_data, tgt_data))
        return DataLoader(dataset, batch_size=None, shuffle=True)


    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader(num_batches=20)

    # 运行消融实验
    results = experiment.run_ablation_study(train_loader, val_loader, epochs=5)

    # 绘制和打印结果
    experiment.plot_results()
    experiment.print_comparison()