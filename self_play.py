# self_play.py
# 自我对弈数据生成
import random
import copy
from functools import partial
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from config import Config, board_size
from data_utils import calc_next_move, board_to_tensor, augment_data
from mcts import MCTS, evaluation_func  # 使用 mcts 中的 evaluation_func
from model import ValueCNN


# 添加 get_calc 实现
def get_calc(model, board):
    device = next(model.parameters()).device
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        value, policy = model.calc(board_tensor)
    return float(value), policy.squeeze(0).cpu().numpy().tolist()


def generate_random_board(model):
    perm = []
    for i in range(board_size):
        for j in range(board_size):
            perm.append((i, j))
    random.shuffle(perm)

    best_val = 1e9
    best_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    if random.randint(0, 4) == 0:
        num_run = 0
    else:
        num_run = random.randint(50, random.randint(50, 1000))

    for t in range(num_run):
        num_moves = random.randint(0, 10)
        board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        current_player = -1
        for _ in range(num_moves):
            i, j = perm[_]
            board[i][j] = current_player
            current_player = -current_player
        if evaluation_func(board) != 0:  # 使用 mcts 中的 evaluation_func
            continue

        value, policy = get_calc(model, board)
        val = max(float(value), -float(value))
        if val < best_val:
            best_val = val
            best_board = board
    return best_board


def generate_single_game(_, model_state_dict, num_simulations):
    """单局自我对弈流程"""
    # 初始化模型
    model = ValueCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # 生成初始棋盘 (带随机性)
    board = generate_random_board(model)
    # 温度参数控制探索程度
    temperature = 0.1 * random.randint(0, 9)

    # 运行MCTS
    with torch.no_grad():
        mcts = MCTS(model)

        root = None # 复用搜索树
        for move_num in range(board_size * board_size):
            # 动态调整探索参数
            if move_num < 10:
                # 添加 dirichlet_alpha 属性
                mcts.dirichlet_alpha = 0.3
            else:
                mcts.dirichlet_alpha = 0.01

            # MCTS模拟获取动作概率
            infos, new_root = mcts.run(board, num_simulations, train=0, cur_root=root, return_root=1)
            value, action_probs = infos

            # 根据温度采样动作
            action = calc_next_move(board, action_probs, temperature)
            board[action[0]][action[1]] = 1

            # 更新搜索树根节点
            if action in new_root.children and new_root.children[action][0] is not None:
                root = new_root.children[action][0]
            else:
                root = None

            # 终局检查
            if mcts.is_terminal(board):
                break

            for i in range(board_size):
                for j in range(board_size):
                    board[i][j] *= -1

        #torch.cuda.empty_cache()
        # 返回训练数据
        return mcts.get_train_data()


def generate_selfplay_data(model, num_games, num_simulations=Config.train_simulation):
    """并行生成自我对弈数据"""
    model_state_dict = model.state_dict()
    mp.set_start_method('spawn', force=True)

    # 使用多进程加速
    with mp.Pool(processes=Config.num_workers) as pool:
        func = partial(
            generate_single_game,
            model_state_dict=model_state_dict,
            num_simulations=num_simulations
        )
        results = list(tqdm(
            pool.imap(func, range(num_games)),
            total=num_games,
            desc="生成自对弈对局数据"
        ))

    # 合并所有对局数据
    boards, policies, values, weights = [], [], [], []
    for game_boards, game_policies, game_values, game_weights in results:
        boards.extend(game_boards)
        policies.extend(game_policies)
        values.extend(game_values)
        weights.extend(game_weights)

    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.FloatTensor(values)
    weights = torch.FloatTensor(weights)

    # 数据增强 (旋转/翻转)
    boards, policies, values, weights = augment_data(boards, policies, values, weights)

    return boards, policies, values, weights