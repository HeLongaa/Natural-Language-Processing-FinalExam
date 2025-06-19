# 任务二：基于深度模型的自然语言推理

本项目实现了基于SNLI数据集的自然语言推理任务，包括三种深度学习模型：
1. GloVe + BiLSTM
2. BERT（词向量层）+ BiLSTM
3. BERT 下游任务精调模型

## 环境设置

运行以下命令安装必要的依赖：

```bash
bash setup.sh
```

## 数据集

Stanford Natural Language Inference (SNLI) 数据集包含前提（premise）和假设（hypothesis）文本对，标签分为三类：
- `0 (Entailment)`：前提可推断假设
- `1 (Neutral)`：其他情况
- `2 (Contradiction)`：前提与假设矛盾

数据集存储在 `data/` 目录下的 parquet 文件中。

## 使用方法

### 训练和评估模型

您可以使用以下命令来训练和评估模型：

```bash
# 训练 GloVe + BiLSTM 模型
python main.py --model glove_bilstm --epochs 10 --batch_size 32 --lr 1e-3 --glove_dim 100

# 训练 BERT + BiLSTM 模型
python main.py --model bert_bilstm --epochs 5 --batch_size 16 --lr 5e-4

# 训练 BERT 微调模型
python main.py --model bert_fine_tuning --epochs 3 --batch_size 16 --lr 1e-5
```

### 参数说明

- `--model`: 选择要训练的模型，可选值为 `glove_bilstm`, `bert_bilstm`, `bert_fine_tuning`
- `--batch_size`: 批量大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--max_length`: 最大序列长度
- `--glove_dim`: GloVe词向量维度，可选值为 50, 100, 200, 300
- `--hidden_size`: LSTM隐藏层大小
- `--num_layers`: LSTM层数
- `--dropout`: Dropout比率
- `--seed`: 随机种子
- `--no_cuda`: 是否禁用CUDA
- `--output_dir`: 输出目录

## 输出文件

训练过程会在 `outputs/` 目录下生成以下文件：
- 模型权重文件 (`.pth`)
- 训练历史图表 (loss和accuracy曲线)
- 评估指标 (`metrics.txt`)

## 评估指标

- 准确率 (Accuracy)
- 宏平均F1值 (Macro-F1)

## 模型结构

### GloVe + BiLSTM
使用预训练的GloVe词向量将文本转换为词嵌入，然后通过双向LSTM处理前提和假设文本，最后使用全连接层进行分类。

### BERT + BiLSTM
使用BERT模型的词向量层提取特征，然后通过双向LSTM进行处理，最后使用全连接层进行分类。

### BERT 微调
直接使用BERT模型，并在[CLS]标记的表示之上添加一个分类层，对整个模型进行微调。
