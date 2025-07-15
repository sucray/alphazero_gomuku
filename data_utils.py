# data_utils.py
# 数据处理工具
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
from config import board_size


def board_to_tensor(board: list):
    """将棋盘状态转换为3通道张量:
        通道0: 当前玩家棋子
        通道1: 对手棋子
        通道2: 空位
        """
    board = np.array(board)
    current_player = (board == 1).astype(np.float32)
    opponent = (board == -1).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    tensor = np.stack([current_player, opponent, empty], axis=0)
    return torch.FloatTensor(tensor)


class Weighted_Dataset(Dataset):
    """带权重的数据集"""
    def __init__(self, boards, policies, values, weights):
        self.boards = boards  # 棋盘状态张量
        self.policies = policies
        self.values = values
        self.weights = weights

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx], self.weights[idx]


def augment_data(boards, policies, values, weights):
    augmented_boards = []
    augmented_policies = []
    augmented_values = []
    augmented_weights = []

    for board, policy, value, weight in zip(boards, policies, values, weights):
        value = torch.tensor(value).clone().detach().float()
        weight = torch.tensor(weight).clone().detach().float()

        for k in range(4):
            for o in range(2):
                new_board = torch.rot90(board, k, [1, 2])
                new_policy = torch.rot90(policy, k, [0, 1])
                if o:
                    new_board = torch.flip(new_board, [2])
                    new_policy = torch.flip(new_policy, [1])

                augmented_boards.append(new_board)
                augmented_policies.append(new_policy)
                augmented_values.append(value)
                augmented_weights.append(weight)

    return (
        torch.stack(augmented_boards),
        torch.stack(augmented_policies),
        torch.stack(augmented_values),
        torch.stack(augmented_weights)
    )


def calc_next_move(board, probs, temperature=0):
    valid_moves = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                valid_moves.append((i, j, probs[i][j]))
    if temperature == 0:
        valid_moves.sort(key=lambda x: x[2], reverse=True)
        return valid_moves[0][:2]
    else:
        moves = [(i, j) for i, j, _ in valid_moves]
        probs = np.array([p for _, _, p in valid_moves], dtype=np.float64)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
        chosen_index = np.random.choice(len(moves), p=probs)
        return moves[chosen_index]