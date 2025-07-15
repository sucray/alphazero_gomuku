# model.py
# 五子棋神经网络模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import board_size


class ResidualBlock(nn.Module):
    """残差块结构 (用于构建深层网络)"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 残差连接结构
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # 跳跃连接
        out = self.relu(out)
        return out


class ValueCNN(nn.Module):
    """五子棋双头评估网络 (策略+价值)"""
    def __init__(self, in_channels=3, hidden_channels=32, num_blocks=5, value_dim=128):
        super(ValueCNN, self).__init__()

        # 输入层
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)

        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # 策略头 (预测落子概率)
        self.policy_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.policy_conv2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)

        # 价值头 (评估局面胜负)
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_dim)
        self.value_fc2 = nn.Linear(value_dim, 1)

    def forward(self, x):
        # 特征提取
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        # 策略输出
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.squeeze(1)
        policy_logits = policy.view(x.size(0), -1)

        # 价值输出
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # 输出范围[-1,1]

        return value, policy_logits

    def calc(self, x):
        """推理接口 (返回概率分布)"""
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(x)
            probs = F.softmax(logits, dim=1).view(-1, board_size, board_size)
            return value, probs