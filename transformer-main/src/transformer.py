import torch
import torch.nn as nn
import math
import copy

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=512, ablation_type="full"):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.ablation_type = ablation_type

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,ablation_type)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,ablation_type)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        self.parameters() 返回模型中所有可训练参数
        包括：权重矩阵、偏置向量、嵌入矩阵等
        为什么只初始化多维参数？
        1D参数（如偏置）通常用零初始化
        多维参数需要更精细的初始化策略
        Returns:
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        if self.ablation_type == "no_positional_encoding":
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        else:
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        output = self.generator(output)

        return output


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


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask, memory_mask)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,ablation_type="full"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ablation_type = ablation_type
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
        if self.ablation_type != "no_residual":
            src = self.dropout1(attn_output)
        else:
            src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        if self.ablation_type != "no_residual":
            src = self.dropout2(ff_output)
        else:
            src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,ablation_type="full"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ablation_type = ablation_type
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
        if self.ablation_type != "no_residual":
            tgt = self.dropout1(attn_output)
        else:
            tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.multihead_attn(tgt, memory, memory, memory_mask)
        if self.ablation_type != "no_residual":
            tgt = self.dropout2(attn_output)
        else:
            tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        if self.ablation_type != "no_residual":
            tgt = self.dropout3(attn_output)
        else:
            tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 确认是否能被多头整除
        assert d_model % nhead == 0
        # 计算每个头的词向量维度，在词向量维度进行拆分
        self.d_k = d_model // nhead

        self.nhead = nhead
        # 根据图示定义四个线性层
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

        # 数据变化[ 2 , 4 , 512 ]  --->   [ 2 , 4 , 8 , 64 ]   --->    [ 2 , 8 , 4 , 64 ]
        #       [batch_size , seq_len , d_model]   ------->     [batch_size , seq_len , multiHead , d_model/heads ]
        #                                          ------->     [batch_size, multiHead, seq_len, d_model / heads]
        # -1表示的意思是让它自己进行计算
        # 必须用转置，转置之后4代表的还是长度，但是用view的话，4代表的就不一定是seq_len了
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        """
        query-->[2,8,4,64],key--->[2,8,4,64]----转置矩阵相乘--->[2,8,4,4]  最后两个维度是矩阵相乘
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            """
            假设原始掩码形状: [batch_size, seq_len]
            mask = torch.tensor([[1, 1, 0, 0]])  # 形状 [1, 4]
            # 注意力分数形状: [batch_size, num_heads, seq_len, seq_len]  
            scores = torch.randn(1, 8, 4, 4)     # 形状 [1, 8, 4, 4]
            # 扩展后掩码形状: [1, 8, 4, 4]
            mask_expanded = mask.expand_as(scores)
            """
            mask_expanded = mask.expand_as(scores)
            scores.masked_fill_(mask_expanded == 0, -1e9)

            # 对注意力分数归一化
        attention = torch.softmax(scores, dim=-1)

        # 随机失活
        attention = self.dropout(attention)

        """
        注意力分数乘V最后的形状仍然是[2,8,4,64]
        """
        output = torch.matmul(attention, V)

        # contiguous()让内存地址连续，只有用transpose在转置的时候才加这个函数，要是使用permit方法就不用了
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        output = self.out_linear(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask