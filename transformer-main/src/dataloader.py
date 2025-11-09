import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
import os
import zipfile

"""
1、对数据集中的信息进行清洗
2、创建源语言和目标语言的词表
3、返回训练集和测试集的dataLoader

     原始文本    →        分词        →    索引化  →    添加特殊标记  →      填充对齐     → 批次数据
        ↓                ↓                ↓             ↓                 ↓            ↓
 "Hello world" → ["Hello", "world"] → [43, 27] → [1, 43, 27, 2] → [1,43,27,2,0,0] → 批次张量
 
"""

class IWSLT2017Dataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=128):
        self.src_sentences = src_sentences  # 源语言句子（德语）
        self.tgt_sentences = tgt_sentences  # 目标语言句子（英语）
        self.src_vocab = src_vocab          # 源语言词汇表
        self.tgt_vocab = tgt_vocab          # 目标语言词汇表
        self.max_length = max_length        # 最大序列长度

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]  # 获取源句子
        tgt = self.tgt_sentences[idx]  # 获取目标句子

        # 源语言分词和索引化
        src_tokens = []
        for token in src.split()[:self.max_length - 2]:  # 限制长度并留出特殊标记位置 句子最大下标之后的位置留一个空位
            if token in self.src_vocab:
                src_tokens.append(self.src_vocab[token])  # 词到索引的映射
            else:
                src_tokens.append(self.src_vocab['<unk>'])  # 未知词处理

        tgt_tokens = []
        for token in tgt.split()[:self.max_length - 2]:
            if token in self.tgt_vocab:
                tgt_tokens.append(self.tgt_vocab[token])
            else:
                tgt_tokens.append(self.tgt_vocab['<unk>'])

        # 添加开始和结束标记
        src_tokens = [self.src_vocab['<sos>']] + src_tokens + [self.src_vocab['<eos>']]
        tgt_tokens = [self.tgt_vocab['<sos>']] + tgt_tokens + [self.tgt_vocab['<eos>']]

        """
        # 输入句子
        tgt_sentence = "I love machine learning"
        src_sentence = "Ich liebe maschinelles Lernen"
        # 处理后
        tgt_tokens = [1, 15, 128, 347, 892, 2]    # <sos>, I, love, machine, learning, <eos>
        src_tokens = [1, 25, 167, 458, 723, 2]    # <sos>, Ich, liebe, maschinelles, Lernen, <eos>
        """
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


"""
构建词表：
"""
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())  # 统计每个词的出现频率

    # 统计单词出现的频率，不一定是按照降序进行排序，而是随机的
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}  # 特殊标记
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:  # 只保留出现频率≥min_freq的词
            vocab[word] = idx
            idx += 1

    return vocab


def load_iwslt_data(data_path, max_samples=10000):
    # 直接指定train_dir为你的de-en目录（根据服务器实际路径修改）
    train_dir = os.path.join(data_path, "de-en")  # 假设data_path是./data，这里就是./data/de-en

    # 检查train_dir下是否有目标文件
    if not (os.path.exists(os.path.join(train_dir, "train.tags.de-en.en")) and
            os.path.exists(os.path.join(train_dir, "train.tags.de-en.de"))):
        raise FileNotFoundError(f"Could not find train files in {train_dir}")

    print(f"Found training data in: {train_dir}")

    # 后续读取文件的代码不变...
    src_file = os.path.join(train_dir, "train.tags.de-en.en")
    tgt_file = os.path.join(train_dir, "train.tags.de-en.de")

    """
    把标签行数据给去掉
    <talk id="TED1">
    <url>http://www.ted.com/talks/...</url>
    <title>My journey to ...</title>
    <description>In this talk ...</description>
    <keywords>culture, travel, story</keywords>
    <speaker>John Smith</speaker>
    """
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = []
        for line in f:
            line = line.strip()   # 作用：移除字符串首尾的空白字符空格 ' '制表符 '\t'  换行符 '\n'  回车符 '\r'
            if line and not line.startswith('<'):
                src_lines.append(line)

    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                tgt_lines.append(line)

    # 数据量对齐
    min_len = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]

    print("长度为len(src_lines)",len(src_lines),"len(tgt_lines)",len(tgt_lines))
    print("选取长度为", max_samples)

    # 取最大数据个数
    if max_samples > 0:
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]

    # 创建词表
    src_vocab = build_vocab(src_lines, min_freq=1)
    tgt_vocab = build_vocab(tgt_lines, min_freq=1)

    # 用于创建反转的字典映射。创建了翻转词表，使得能够从索引取到单词，之前是单词对应的索引进行嵌入
    src_vocab_inv = {v: k for k, v in src_vocab.items()}
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}

    return src_lines, tgt_lines, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv


"""
得到训练集dataLoader和测试集dataLoader，得到源语言和目标语言的词表，然后得到对应的反转词表
"""
def get_dataloaders(data_path, batch_size=16, max_samples=5000, max_length=64):  # 调整默认值以适应更小的数据集
    src_lines, tgt_lines, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv = load_iwslt_data(data_path, max_samples)

    train_size = int(0.9 * len(src_lines))
    val_size = len(src_lines) - train_size

    train_src = src_lines[:train_size]
    train_tgt = tgt_lines[:train_size]
    val_src = src_lines[train_size:]
    val_tgt = tgt_lines[train_size:]

    train_dataset = IWSLT2017Dataset(train_src, train_tgt, src_vocab, tgt_vocab, max_length)
    val_dataset = IWSLT2017Dataset(val_src, val_tgt, src_vocab, tgt_vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, src_vocab, tgt_vocab, src_vocab_inv, tgt_vocab_inv


"""
数据加载和批处理
"""
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)# zip()将多个可迭代对象"压缩"成元组的迭代器 * 解包操作符可迭代对象解包为单独的参数，先解包再压缩

    # 找到批次中最长的序列
    src_max_len = max([len(seq) for seq in src_batch])
    tgt_max_len = max([len(seq) for seq in tgt_batch])

    src_padded = []
    tgt_padded = []

    # 对每个序列进行填充
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded_tensor = torch.zeros(src_max_len, dtype=torch.long)  # 创建全0张量
        tgt_padded_tensor = torch.zeros(tgt_max_len, dtype=torch.long)

        src_padded_tensor[:len(src)] = src  # 将实际数据复制到前面
        tgt_padded_tensor[:len(tgt)] = tgt

        src_padded.append(src_padded_tensor)
        tgt_padded.append(tgt_padded_tensor)

    """
    批次中不同长度的序列：
    序列1: [1, 43, 27, 2]          → 长度: 4
    序列2: [1, 15, 128, 347, 892, 2] → 长度: 6
    
    填充对齐后：
    序列1: [1, 43, 27, 2, 0, 0]
    序列2: [1, 15, 128, 347, 892, 2]
    
    堆叠成批次张量：
    [[1, 43, 27, 2, 0, 0],
     [1, 15, 128, 347, 892, 2]]
    """
    return torch.stack(src_padded), torch.stack(tgt_padded)