import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class GloveBiLSTM(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_size=256, num_layers=6, dropout=0.5, num_classes=10):
        """
        GloVe + BiLSTM 模型
        :param pretrained_embeddings: 预训练的GloVe词向量
        :param hidden_size: LSTM隐藏层大小
        :param num_layers: LSTM层数
        :param dropout: Dropout比率
        :param num_classes: 分类数量
        """
        super(GloveBiLSTM, self).__init__()
        
        # 词嵌入层
        vocab_size, embed_dim = pretrained_embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2表示双向
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入序列 [batch_size, seq_len]
        :return: 分类结果 [batch_size, num_classes]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # BiLSTM
        output, (hidden, _) = self.lstm(embedded)  # output: [batch_size, seq_len, hidden_size*2]
        
        # 获取最后一个时间步的隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_size*2]
        hidden = self.dropout(hidden)
        
        # 分类
        logits = self.fc(hidden)  # [batch_size, num_classes]
        
        return logits

class BertBiLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.5, num_classes=10, bert_model='bert-base-uncased'):
        """
        BERT + BiLSTM 模型
        :param hidden_size: LSTM隐藏层大小
        :param num_layers: LSTM层数
        :param dropout: Dropout比率
        :param num_classes: 分类数量
        :param bert_model: BERT模型名称或路径
        """
        super(BertBiLSTM, self).__init__()
        
        # BERT模型（只使用词向量层）
        self.bert = BertModel.from_pretrained(bert_model)
        
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2表示双向
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        :param input_ids: 输入ID [batch_size, seq_len]
        :param attention_mask: 注意力掩码 [batch_size, seq_len]
        :param token_type_ids: token类型ID [batch_size, seq_len]
        :return: 分类结果 [batch_size, num_classes]
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # 使用BERT最后一层的隐藏状态
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden_size]
        
        # BiLSTM
        output, (hidden, _) = self.lstm(sequence_output)  # output: [batch_size, seq_len, hidden_size*2]
        
        # 获取最后一个时间步的隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_size*2]
        hidden = self.dropout(hidden)
        
        # 分类
        logits = self.fc(hidden)  # [batch_size, num_classes]
        
        return logits

class BertFineTuning(nn.Module):
    def __init__(self, dropout=0.1, num_classes=10, bert_model='bert-base-uncased'):
        """
        BERT微调模型
        :param dropout: Dropout比率
        :param num_classes: 分类数量
        :param bert_model: BERT模型名称或路径
        """
        super(BertFineTuning, self).__init__()
        
        # BERT模型
        self.bert = BertModel.from_pretrained(bert_model)
        
        # 分类层
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        :param input_ids: 输入ID [batch_size, seq_len]
        :param attention_mask: 注意力掩码 [batch_size, seq_len]
        :param token_type_ids: token类型ID [batch_size, seq_len]
        :return: 分类结果 [batch_size, num_classes]
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]标记的表示进行分类
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
