import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import time
from datetime import datetime

from data_loader import get_dataloaders, load_glove_embeddings
from models import GloveBiLSTM, BertBiLSTM, BertFineTuning
from train import train_model
from evaluate import evaluate_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_history(history, model_name, save_dir='./plots'):
    """
    绘制训练历史
    :param history: 训练历史
    :param model_name: 模型名称
    :param save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制损失
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['dev_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'))
    
    # 绘制准确率
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['dev_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'))

def main():
    parser = argparse.ArgumentParser(description='SNLI 自然语言推理任务')
    parser.add_argument('--model', type=str, default='glove_bilstm', 
                        choices=['glove_bilstm', 'bert_bilstm', 'bert_fine_tuning'],
                        help='要训练的模型 (default: glove_bilstm)')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小 (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数 (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率 (default: 1e-3)')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度 (default: 128)')
    parser.add_argument('--glove_dim', type=int, default=100, 
                        choices=[50, 100, 200, 300], help='GloVe词向量维度 (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM隐藏层大小 (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4, help='LSTM层数 (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比率 (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (default: 42)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='禁用CUDA')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录 (default: ./outputs)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/{args.model}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"使用设备: {device}")

    # 数据路径
    train_path = 'data/train-00000-of-00001.parquet'
    dev_path = 'data/validation-00000-of-00001.parquet'
    test_path = 'data/test-00000-of-00001.parquet'
    
    # 模型训练与评估
    if args.model == 'glove_bilstm':
        # GloVe + BiLSTM
        logger.info("训练 GloVe + BiLSTM 模型")
        
        # 获取数据加载器
        train_loader, dev_loader, test_loader, word_to_idx = get_dataloaders(
            train_path, dev_path, test_path, 
            batch_size=args.batch_size, 
            max_length=args.max_length, 
            use_bert=False
        )
        
        # 加载GloVe词向量
        glove_path = f'../task1/glove/glove.6B.{args.glove_dim}d.txt'
        logger.info(f"从 {glove_path} 加载GloVe词向量")
        embeddings = load_glove_embeddings(glove_path, word_to_idx, embed_dim=args.glove_dim)
        
        # 创建模型
        model = GloveBiLSTM(
            pretrained_embeddings=embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=3  # 0: entailment, 1: neutral, 2: contradiction
        ).to(device)
        
        # 训练模型
        model_save_path = os.path.join(output_dir, 'glove_bilstm_best.pth')
        history = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_save_path=model_save_path
        )
        
        # 绘制训练历史
        plot_history(history, 'GloVe_BiLSTM', save_dir=output_dir)
        
        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(model_save_path))
        metrics = evaluate_model(model, test_loader, device, use_bert=False)
        
    elif args.model == 'bert_bilstm':
        # BERT + BiLSTM
        logger.info("训练 BERT + BiLSTM 模型")
        
        # 获取数据加载器
        train_loader, dev_loader, test_loader, _ = get_dataloaders(
            train_path, dev_path, test_path, 
            batch_size=args.batch_size, 
            max_length=args.max_length, 
            use_bert=True
        )
        
        # 创建模型
        model = BertBiLSTM(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=3,  # 0: entailment, 1: neutral, 2: contradiction
            bert_model='../task1/bert-base-uncased'
        ).to(device)
        
        # 训练模型
        model_save_path = os.path.join(output_dir, 'bert_bilstm_best.pth')
        history = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_save_path=model_save_path,
            use_bert=True
        )
        
        # 绘制训练历史
        plot_history(history, 'BERT_BiLSTM', save_dir=output_dir)
        
        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(model_save_path))
        metrics = evaluate_model(model, test_loader, device, use_bert=True)
        
    elif args.model == 'bert_fine_tuning':
        # BERT 微调
        logger.info("训练 BERT 微调模型")
        
        # 获取数据加载器
        train_loader, dev_loader, test_loader, _ = get_dataloaders(
            train_path, dev_path, test_path, 
            batch_size=args.batch_size, 
            max_length=args.max_length, 
            use_bert=True
        )
        
        # 创建模型
        model = BertFineTuning(
            dropout=args.dropout,
            num_classes=3,  # 0: entailment, 1: neutral, 2: contradiction
            bert_model='../task1/bert-base-uncased'
        ).to(device)
        
        # 训练模型
        model_save_path = os.path.join(output_dir, 'bert_fine_tuning_best.pth')
        history = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=1e-5,  # 使用较小的学习率进行微调
            model_save_path=model_save_path,
            use_bert=True,
            bert_fine_tuning=True
        )
        
        # 绘制训练历史
        plot_history(history, 'BERT_Fine_Tuning', save_dir=output_dir)
        
        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(model_save_path))
        metrics = evaluate_model(model, test_loader, device, use_bert=True)
    
    # 保存评估指标
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"结果已保存到 {output_dir}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"总执行时间: {end_time - start_time:.2f} 秒")
