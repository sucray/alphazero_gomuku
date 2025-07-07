# -*- coding: utf-8 -*-
"""
基于项目一残差网络的AlphaZero策略价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """设置学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ResidualBlock(nn.Module):
    """残差块 (来自项目一)"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class PolicyValueNet(nn.Module):
    """策略价值网络 (基于项目一架构)"""

    def __init__(self, board_width, board_height, in_channels=4, hidden_channels=64, num_blocks=5):
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # 输入层
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)

        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_width * board_height, board_width * board_height)

        # 价值头
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 公共特征提取
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        x_policy = F.relu(self.policy_bn(self.policy_conv(x)))
        x_policy = x_policy.view(-1, 2 * self.board_width * self.board_height)
        x_policy = F.log_softmax(self.policy_fc(x_policy), dim=1)

        # 价值头
        x_value = F.relu(self.value_bn(self.value_conv(x)))
        x_value = x_value.view(-1, self.board_width * self.board_height)
        x_value = F.relu(self.value_fc1(x_value))
        x_value = torch.tanh(self.value_fc2(x_value))

        return x_policy, x_value


class PolicyValueNetWrapper():
    """网络包装类 (兼容原项目二接口)"""

    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2正则系数

        # 使用残差网络
        self.policy_value_net = PolicyValueNet(board_width, board_height)
        if self.use_gpu:
            self.policy_value_net.cuda()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """输入批量状态，返回动作概率和状态价值"""
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.stack(state_batch)).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(np.stack(state_batch)))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """输入棋盘状态，返回可用动作及其概率"""
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 4, self.board_width, self.board_height))

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.data[0][0]

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """训练步骤"""
        state_batch = np.ascontiguousarray(np.stack(state_batch))
        mcts_probs = np.ascontiguousarray(np.stack(mcts_probs))
        winner_batch = np.ascontiguousarray(winner_batch)

        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """保存模型"""
        net_params = self.policy_value_net.state_dict()
        torch.save(net_params, model_file)