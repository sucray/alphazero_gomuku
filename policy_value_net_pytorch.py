# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # Common layers: increased channels and depth
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 32 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New layer
        # Action policy layers
        self.act_conv1 = nn.Conv2d(256, 8, kernel_size=1)  # 4 -> 8
        self.act_fc1 = nn.Linear(8 * board_width * board_height,
                                 board_width * board_height)
        # State value layers
        self.val_conv1 = nn.Conv2d(256, 4, kernel_size=1)  # 2 -> 4
        self.val_fc1 = nn.Linear(4 * board_width * board_height, 128)  # 64 -> 128
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # Common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Additional layer
        # Action policy head
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 8 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # State value head
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 4 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        self.policy_value_net = Net(board_width, board_height).cuda()  # Force GPU
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    lr=1e-3,
                                    weight_decay=self.l2_const)

        if model_file and model_file.endswith('.checkpoint'):
            checkpoint = torch.load(model_file)
            self.policy_value_net.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        state_tensor = torch.from_numpy(current_state).float().cuda()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            value = value.item()
        return list(zip(legal_positions, act_probs[legal_positions])), value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).cuda()
        winner_batch = torch.FloatTensor(winner_batch).cuda()

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

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
