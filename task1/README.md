# IMDB-10 情感分类任务

本项目是自然语言处理课程的任务一：基于深度模型的情感分析。使用IMDB-10数据集进行十分类情感分类任务。

## 项目结构

```
task1/
├── data/                # 数据集目录
│   ├── imdb.train.txt.ss  # 训练集
│   ├── imdb.dev.txt.ss    # 验证集
│   ├── imdb.test.txt.ss   # 测试集
├── glove/               # GloVe词向量目录
│   ├── glove.6B.100d.txt  # 100维GloVe词向量
│   ├── glove.6B.200d.txt  # 200维GloVe词向量
│   ├── glove.6B.300d.txt  # 300维GloVe词向量
│   └── glove.6B.50d.txt   # 50维GloVe词向量
├── outputs/             # 输出目录（模型、图表等）
├── data_loader.py       # 数据加载模块
├── models.py            # 模型定义模块
├── train.py             # 模型训练模块
├── evaluate.py          # 模型评估模块
├── main.py              # 主程序
├── setup.sh             # 环境设置脚本
└── README.md            # 项目说明
```

## 数据集说明

IMDB-10 是电影评论的十分类情感（评分 1-10）标注的数据集。数据格式如下：

```
用户ID \t 产品ID \t 评分(1-10) \t 文本内容（句子间以<sssss>分隔）
```

## 模型说明

本项目实现了三种深度学习模型：

1. **GloVe + BiLSTM**：使用预训练的GloVe词向量和双向LSTM进行情感分类
2. **BERT（词向量层）+ BiLSTM**：使用BERT的词向量层作为特征提取器，配合双向LSTM进行情感分类
3. **BERT 下游任务精调模型**：对BERT进行微调，用于情感分类任务

## 环境设置

运行以下命令安装所需依赖：

```bash
bash setup.sh
```

## 运行指南

### 1. GloVe + BiLSTM 模型

```bash
python main.py --model glove_bilstm --batch_size 16 --epochs 10 --glove_dim 100 --lr 0.00001
```

### 2. BERT + BiLSTM 模型

```bash
python main.py --model bert_bilstm --batch_size 32 --epochs 10 --lr 0.00001
```

### 3. BERT 微调模型

```bash
python main.py --model bert_fine_tuning --batch_size 8 --epochs 3 --lr 2e-5
```

## 参数说明

- `--model`：选择模型，可选 'glove_bilstm', 'bert_bilstm', 'bert_fine_tuning'
- `--batch_size`：批处理大小，默认32
- `--epochs`：训练轮数，默认10
- `--lr`：学习率，默认1e-3
- `--max_length`：最大序列长度，默认512
- `--glove_dim`：GloVe词向量维度，可选 50, 100, 200, 300，默认100
- `--hidden_size`：LSTM隐藏层大小，默认128
- `--num_layers`：LSTM层数，默认3
- `--dropout`：Dropout比率，默认0.5
- `--seed`：随机种子，默认42
- `--no_cuda`：禁用CUDA
- `--output_dir`：输出目录，默认'./outputs'

## 评价指标

- **F1 值（macro-F1）**：主要评价指标
- **准确度（Accuracy）**
- **均方根误差（Root Mean Squared Error, RMSE）**
