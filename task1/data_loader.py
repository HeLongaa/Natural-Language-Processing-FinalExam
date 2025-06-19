import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer

class IMDBDataset(Dataset):
    def __init__(self, file_path, max_length=512, tokenizer=None, use_bert=False):
        """
        IMDB-10数据集加载
        :param file_path: 数据文件路径
        :param max_length: 最大序列长度
        :param tokenizer: 词向量转换器
        :param use_bert: 是否使用BERT
        """
        self.data = []
        self.labels = []
        self.max_length = max_length
        self.use_bert = use_bert
        self.tokenizer = tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用多个连续的空格或制表符作为分隔符
                parts = [part.strip() for part in line.strip().split() if part.strip()]
                
                # 确保至少有4个字段（用户ID，产品ID，评分，文本内容）
                if len(parts) >= 4:
                    # 前两个字段是用户ID和产品ID，第三个字段是评分
                    user_id = parts[0]
                    product_id = parts[1]
                    
                    try:
                        # 尝试获取评分标签(1-10)
                        label = int(parts[2])
                        
                        # 获取文本内容（剩余的所有部分）
                        text = ' '.join(parts[3:])
                        
                        # 过滤掉<sssss>分隔符
                        text = text.replace(' <sssss> ', ' ')
                        
                        self.data.append(text)
                        self.labels.append(label - 1)  # 将标签转换为0-9范围
                    except ValueError:
                        # 如果评分不是有效的整数，则跳过该样本
                        continue
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        
        if self.use_bert:
            # 使用BERT tokenizer处理文本
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros(self.max_length, dtype=torch.long)).flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # 返回原始文本和标签
            return text, label

def load_glove_embeddings(glove_path, word_to_idx, embed_dim=100):
    """
    加载GloVe词向量
    :param glove_path: GloVe文件路径
    :param word_to_idx: 词到索引的映射
    :param embed_dim: 词向量维度
    :return: 词向量矩阵
    """
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embed_dim))
    embeddings[0] = np.zeros(embed_dim)  # 为填充token设置零向量
    
    # 读取GloVe词向量
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_idx:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word_to_idx[word]] = vector
    
    return torch.FloatTensor(embeddings)

def build_vocab(data_path, min_freq=5):
    """
    构建词汇表
    :param data_path: 数据文件路径
    :param min_freq: 最小词频
    :return: 词到索引的映射，索引到词的映射
    """
    word_counts = {}
    
    # 统计词频
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用多个连续的空格或制表符作为分隔符
            parts = [part.strip() for part in line.strip().split() if part.strip()]
            
            # 确保至少有4个字段（用户ID，产品ID，评分，文本内容）
            if len(parts) >= 4:
                try:
                    # 尝试获取评分标签(1-10)
                    label = int(parts[2])
                    
                    # 获取文本内容（剩余的所有部分）
                    text = ' '.join(parts[3:])
                    text = text.replace(' <sssss> ', ' ')
                    
                    for word in text.split():
                        word_counts[word] = word_counts.get(word, 0) + 1
                except ValueError:
                    # 如果评分不是有效的整数，则跳过该样本
                    continue
    
    # 构建词汇表
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

def text_to_sequence(text, word_to_idx, max_length=512):
    """
    将文本转换为索引序列
    :param text: 输入文本
    :param word_to_idx: 词到索引的映射
    :param max_length: 最大序列长度
    :return: 索引序列
    """
    words = text.split()
    sequence = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words[:max_length]]
    
    # 填充或截断
    if len(sequence) < max_length:
        sequence += [word_to_idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return torch.tensor(sequence, dtype=torch.long)

class IMDBCollator:
    def __init__(self, word_to_idx, max_length=512):
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __call__(self, batch):
        texts, labels = zip(*batch)
        sequences = [text_to_sequence(text, self.word_to_idx, self.max_length) for text in texts]
        labels = torch.tensor(labels, dtype=torch.long)
        return torch.stack(sequences), labels

def get_dataloaders(train_path, dev_path, test_path, batch_size=32, max_length=512, use_bert=False):
    """
    获取数据加载器
    :param train_path: 训练集路径
    :param dev_path: 验证集路径
    :param test_path: 测试集路径
    :param batch_size: 批量大小
    :param max_length: 最大序列长度
    :param use_bert: 是否使用BERT
    :return: 训练集、验证集和测试集的数据加载器
    """
    if use_bert:
        # 使用BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        train_dataset = IMDBDataset(train_path, max_length, tokenizer, use_bert=True)
        dev_dataset = IMDBDataset(dev_path, max_length, tokenizer, use_bert=True)
        test_dataset = IMDBDataset(test_path, max_length, tokenizer, use_bert=True)
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            token_type_ids = torch.stack([item['token_type_ids'] for item in batch]) if 'token_type_ids' in batch[0] else None
            labels = torch.stack([item['label'] for item in batch])
            
            if token_type_ids is not None:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': labels
                }
            else:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    else:
        # 使用GloVe词向量
        word_to_idx, _ = build_vocab(train_path)
        collator = IMDBCollator(word_to_idx, max_length)
        
        train_dataset = IMDBDataset(train_path, max_length)
        dev_dataset = IMDBDataset(dev_path, max_length)
        test_dataset = IMDBDataset(test_path, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    
    return train_loader, dev_loader, test_loader, word_to_idx if not use_bert else None
