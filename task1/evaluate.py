import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import math
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device, use_bert=False):
    """
    评估模型
    :param model: 模型
    :param test_loader: 测试数据加载器
    :param device: 设备
    :param use_bert: 是否使用BERT
    :return: 评估指标
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 准备数据
            if use_bert:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            
            # 保存预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    
    metrics = {
        'accuracy': accuracy * 100,  # 转换为百分比
        'macro_f1': macro_f1,
        'rmse': rmse
    }
    
    # 打印评估结果
    logger.info(f"Evaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"Macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    
    return metrics
