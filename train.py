# train.py
# 模型训练逻辑
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from data_utils import Weighted_Dataset

def train_model(model, train_loader, val_loader, config):
    """单轮模型训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数
    value_criterion = nn.MSELoss(reduction='none') # 价值损失
    policy_criterion = nn.KLDivLoss(reduction='none') # 策略损失

    val_value_criterion = nn.MSELoss()
    val_policy_criterion = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(config.num_epochs):
        model.train()
        train_value_loss, train_policy_loss = 0, 0

        # 训练批次循环
        for batch_boards, batch_policies, batch_values, batch_weights in tqdm(train_loader,
                                                                              desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(batch_policies.size(0), -1)
            batch_values = batch_values.to(device).unsqueeze(1)
            batch_weights = batch_weights.to(device)

            # 前向传播
            optimizer.zero_grad()
            pred_values, pred_policies = model(batch_boards)

            # 计算加权损失
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)
            policy_loss = policy_criterion(F.log_softmax(pred_policies, dim=1),
                                           batch_policies.view(-1, batch_policies.size(-1))).sum(dim=1, keepdim=True)

            weighted_value_loss = (value_loss * batch_weights).mean()
            weighted_policy_loss = (policy_loss * batch_weights).mean()

            # 组合损失 (价值:策略=2:1)
            loss = 2 * weighted_value_loss + weighted_policy_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录损失
            train_value_loss += weighted_value_loss.item()
            train_policy_loss += weighted_policy_loss.item()

        # 验证阶段
        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values, weights in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(policies.size(0), -1)
                values = values.to(device).unsqueeze(1)

                pred_values, pred_policies = model(boards)
                val_value_loss += val_value_criterion(pred_values, values).item()
                val_policy_loss += val_policy_criterion(
                    F.log_softmax(pred_policies, dim=1),
                    policies.view(-1, policies.size(-1))
                ).item()

        avg_train_value = train_value_loss / len(train_loader)
        avg_train_policy = train_policy_loss / len(train_loader)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)

        val_total_loss = 2 * avg_val_value + avg_val_policy
        scheduler.step(val_total_loss)

        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(avg_val_value + avg_val_policy)

    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()