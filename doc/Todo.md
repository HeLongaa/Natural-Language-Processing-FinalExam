# 2025年春季《自然语言处理》期末综合设计

## 一、基于深度模型的情感分析和逻辑推理任务

### 1. 任务一：情感分类
**(1) 任务介绍**：基于 IMDB-10 数据集构建深度模型实现十分类情感分类。  
**(2) 数据集**：IMDB-10 是电影评论的十分类情感（评分 1-10）标注的数据集。  
**(3) 评价指标**：F1 值（macro-F1，主要）、准确度（Accuracy）、均方根误差（Root Mean Squared Error, RMSE）  

#### (4) 数据样本
| Text | Label |
|------|-------|
| i excepted a lot from this movie , and it did deliver . <sssss> there is some great buddhist wisdom in this movie .<sssss> the real dalai lama is a very interesting person , and i think there is a lot of wisdom in buddhism .<sssss> the music , of course , sounds like because it is by philip glass .<sssss> this adds to the beauty of the movie .<sssss> whereas other biographies of famous people tend to get very poor this movie always stays focused and gives a good and honest portrayal of the dalai lama .<sssss> all things being equal , it is a great movie , and i really enjoyed it .<sssss> it is not like taxi driver of course but as a biography of a famous person it is really a great film indeed . | 10 |
| i agree , i also do not like paris , i was there and it was dirty , criminal , hasty and expensive .<sssss> the film tries its best to show paris in its best colors .<sssss> out of 16 episodes , the half is terrible weak , dull , boring and largely uninteresting .<sssss> often , they lack cohesion and they seem to go simply nowhere .<sssss> the worst ones are the zany vampire ditty with elijah wood and several more .<sssss> the one with nick nolte begs for continuation but fails too .<sssss> the other half is a winner , the parts with natale portman , sieve buscemi and then bob hoskins are brilliant and great ... and then there is the best one about the poor killed african american guitar man who was a victim of the white gang .<sssss> this short episode stand out and is the gem alone . | 5 |
| crouching tiger , hidden dragon is such a bad movie .<sssss> the love stories don ’ t make sense at all .<sssss> the cute princess is so annoying .<sssss> the violence does not make any sense either .<sssss> all in all a perfectly pointless movie .<sssss> why should we care about these stupid characters who act in such meaningless ways ? | 1 |

> **提示**  
> - 该数据集中，大部分文本长度超过 512 个 Token，可只取前 512 个 Token 作为输入  
> - `【<sssss>】`是句子分隔符，需过滤  
> - 数据集中含用户 ID 和产品 ID 列，需忽略  

---

### 2. 任务二：蕴含识别
**(1) 任务介绍**：基于 SNLI 数据集构建深度模型实现自然语言推理（识别文本蕴含关系）。  
**(2) 数据集**：Stanford Natural Language Inference (SNLI)，标签分为三类：  
- `0 (Entailment)`：前提可推断假设  
- `2 (Contradiction)`：前提与假设矛盾  
- `1 (Neutral)`：其他情况  
**(3) 评价指标**：F1 值（macro-F1，主要）、准确度（Accuracy）  

#### (4) 数据样本
| Text 1 (Premise) | Text 2 (Hypothesis) | Label |
|------------------|---------------------|-------|
| A person on a horse jumps over a broken down airplane. | A person is training his horse for a competition. | 1 |
| A person on a horse jumps over a broken down airplane. | A person is at a diner, ordering an omelette. | 2 |
| A person on a horse jumps over a broken down airplane. | A person is outdoors, on a horse. | 0 |

> **提示**  
> - Text 1 和 Text 2 存在顺序约束  
> - 数据集为 parquet 格式，可从 [HuggingFace](https://huggingface.co/datasets/stanfordnlp/snli) 下载  

---

## 二、总体要求
1. **任务内容**：  
   - 使用 PyTorch 完成两个任务（数据下载/加载、词向量加载、模型搭建/训练/推理）  
   - 实现以下模型：  
     (1) GloVe + BiLSTM  
     (2) BERT（词向量层）+ BiLSTM  
     (3) BERT 下游任务精调模型  

2. **提交材料**：  
   - 模型结构图、实验结果截图  
   - 源代码（`*.py`）  
   - 电子实验报告（`姓名_学号_实验报告.pdf`）  
   - 打包为 `姓名_学号_期末考核.zip/rar` 提交至雨课堂  
   - **纸质报告**于 **2025年6月30日** 前交至周洲有同学（信息学院1518室）  

3. **数据集划分**：  
   - 训练集：优化网络  
   - 验证集：超参选取  
   - 测试集：结果预测  

4. **实验总结**：分析问题与收获  

---

## 三、评分依据（100分）
| 项目 | 占比 |
|------|------|
| 系统正确性 | 10% |
| 功能完整性 | 40% |
| 代码规范性（缩进/大小写） | 10% |
| 结果正确性 | 30% |
| 报告排版与书写质量 | 10% |