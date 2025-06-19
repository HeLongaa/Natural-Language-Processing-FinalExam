import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, train_loader, dev_loader, device, num_epochs=10, learning_rate=1e-3, 
                model_save_path=None, use_bert=False, bert_fine_tuning=False):
    """
    训练模型
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param dev_loader: 验证数据加载器
    :param device: 设备
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param model_save_path: 模型保存路径
    :param use_bert: 是否使用BERT
    :param bert_fine_tuning: 是否微调BERT
    :return: 训练历史
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if bert_fine_tuning:
        # 微调BERT，使用不同的学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    else:
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }
    
    # 最佳模型保存
    best_dev_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练阶段
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_bar:
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
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({'loss': train_loss / (train_bar.n + 1), 
                                   'acc': 100. * train_correct / train_total})
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            dev_bar = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for batch in dev_bar:
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
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 统计
                dev_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                dev_total += labels.size(0)
                dev_correct += (predicted == labels).sum().item()
                
                # 保存预测和真实标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                dev_bar.set_postfix({'loss': dev_loss / (dev_bar.n + 1), 
                                     'acc': 100. * dev_correct / dev_total})
        
        # 计算验证指标
        dev_loss = dev_loss / len(dev_loader)
        dev_acc = 100. * dev_correct / dev_total
        
        # 更新学习率
        scheduler.step(dev_loss)
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)
        
        # 打印训练信息
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Valid Loss: {dev_loss:.4f}, Valid Acc: {dev_acc:.2f}%")
        
        # 保存最佳模型
        if dev_loss < best_dev_loss and model_save_path:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")
    
    return history
