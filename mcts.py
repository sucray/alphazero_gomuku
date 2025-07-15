# mcts.py
# 蒙特卡洛树搜索实现
import math
import copy
import random
import time
import torch
import numpy as np
from config import Config, board_size
from model import ValueCNN
from data_utils import board_to_tensor



def evaluation_func(board: list):
    num_used = 0
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != 0:
                num_used += 1
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != 0:
                for (x, y) in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cnt = 0
                    for d in range(5):
                        ni = i + d * x
                        nj = j + d * y
                        if 0 <= ni < board_size and 0 <= nj < board_size and board[i][j] == board[ni][nj]:
                            cnt += 1
                        else:
                            break
                    if cnt == 5:
                        if board[i][j] == 1:
                            return 1 - num_used * 3e-4
                        else:
                            return -(1 - num_used * 3e-4)
    return 0


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board # 当前棋盘状态
        self.parent = parent # 父节点
        self.move = move # 到达本节点的动作
        self.children = {} # 子节点字典 {move: (node, prior)}
        self.visit_count = 0 # 访问次数
        self.value_sum = 0 # 价值累加
        self.val = 0 # 节点价值评估
        self.value = None # 原始网络评估值

    def update_value(self):
        """更新节点价值评估"""
        if len(self.children) == 0:
            self.val = self.value
        else:
            if Config.mcts_type == 'mean':
                self.val = self.value_sum / self.visit_count
            else:  # max模式
                vals = max((-child.val if child != None else -10.) for child, prior in self.children.values())
                self.val = self.value if vals == -10 else vals


accumulate_sum = 0


class MCTS:
    def __init__(self, model, c_puct=0.8, puct2=0.02, parallel=0, use_rand=0.01):
        self.c_puct = c_puct # 探索系数
        self.puct2 = puct2   # 二级探索系数
        self.use_rand = use_rand
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if type(model) == str:
            self.model = ValueCNN()
            self.model.load_state_dict(torch.load(model, map_location=self.device, weights_only=True))
        else:
            self.model = model.to(self.device)
        self.visited_nodes = []  # 记录访问节点(用于训练数据收集)

    def new_node(self, *args, **kwargs):
        node = MCTSNode(*args, **kwargs)
        if Config.collect_subnode:
            self.visited_nodes.append(node)
        return node

    def no_child(self, board):
        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] == 0:
                    return False
        return True

    def is_terminal(self, board):
        if self.no_child(board):
            return True
        return evaluation_func(board) != 0

    def run(self, root_board, num_simulations, train=0, cur_root=None, return_root=0):
        global accumulate_sum
        accumulate_sum = 0
        all_beg = time.time_ns()

        if cur_root == None:
            root = MCTSNode(root_board)
            self.visited_nodes.append(root)
        else:
            root = cur_root

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            while node.children:
                node = self.select_child(node)
                search_path.append(node)

            if not self.is_terminal(node.board):
                self.expand_node(node)
            else:
                node.value = self.evaluate_node(node)

            value = self.evaluate_node(node)

            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                node.update_value()
                value = -value

        if not return_root:
            return self.get_results(root, train=train)
        else:
            return self.get_results(root, train=train), root

    def select_child(self, node: MCTSNode):
        """选择最优子节点 (UCT算法)"""
        global accumulate_sum
        beg = time.time_ns()
        total_visits = sum((child.visit_count if child != None else 0) for child, prior in node.children.values())
        explore_buff = math.pow(total_visits + 1, 0.5)
        log_total = math.log(total_visits + 1)

        best_score = -1e9
        best_move = None

        # 计算子节点平均价值 (用于探索平衡)
        exp1 = 0
        exp2 = 0
        for child, prior in node.children.values():
            if child != None:
                exp1 += child.val * child.visit_count
                exp2 += child.visit_count
        ave = exp1 / (exp2 + 1e-5)

        for move, (child, prior) in node.children.items():
            # UCT计算公式
            explore = self.c_puct * prior * explore_buff
            exploit = ave
            if child != None and child.visit_count != 0:
                exploit = child.val
                explore /= (child.visit_count + 1)

            explore += self.puct2 * math.sqrt(log_total / ((child.visit_count if child else 0) + 1))
            score = explore - exploit
            if score > best_score:
                best_score = score
                best_move = move
        accumulate_sum += time.time_ns() - beg

        # 扩展未访问的子节点
        chd, pri = node.children[best_move]
        if chd == None:
            i, j = best_move
            new_board = copy.deepcopy(node.board)
            new_board[i][j] = 1
            for x in range(board_size):
                for y in range(board_size):
                    new_board[x][y] *= -1
            chd = MCTSNode(new_board, parent=node, move=best_move)
            if Config.collect_subnode:
                self.visited_nodes.append(chd)
            node.children[best_move] = (chd, pri)
        return chd

    def expand_node(self, node: MCTSNode):
        """扩展叶子节点"""
        # 使用神经网络预测初始策略和价值
        global accumulate_sum
        tm_beg = time.time_ns()

        # 添加 get_calc 实现
        def get_calc(model, board):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
            with torch.no_grad():
                value, policy = model.calc(board_tensor)
            return float(value), policy.squeeze(0).cpu().numpy().tolist()

        node.value, policy = get_calc(self.model, node.board)

        sum_1 = 0
        valid_moves = []
        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:
                    sum_1 += policy[i][j]
                    valid_moves.append((i, j))
        if sum_1 == 0:
            sum_1 = 1e-10

        # 添加Dirichlet噪声 (根节点专用)
        if node.parent is None and len(valid_moves) > 0:
            noise = np.random.dirichlet([Config.dirichlet_alpha] * len(valid_moves))
            for idx, (i, j) in enumerate(valid_moves):
                policy[i][j] = (1 - Config.dirichlet_epsilon) * policy[i][j] + \
                               Config.dirichlet_epsilon * noise[idx]

        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:
                    node.children[(i, j)] = (None, policy[i][j] / sum_1 + \
                                             random.normalvariate(mu=0, sigma=self.use_rand))

        accumulate_sum += time.time_ns() - tm_beg

    def evaluate_node(self, node: MCTSNode):
        if self.no_child(node.board):
            return 0
        eval = evaluation_func(node.board)
        if eval != 0:
            return eval
        if node.value == None:
            assert False
        return node.value

    def get_results(self, root: MCTSNode, train=0):
        probs = np.zeros((board_size, board_size))
        total_visits = sum((child.visit_count if child != None else 0) for child, prior in root.children.values())

        if not train:
            for move, (child, prior) in root.children.items():
                if child != None:
                    i, j = move
                    probs[i, j] = child.visit_count / total_visits if total_visits > 0 else 0
            return root.value_sum / root.visit_count, probs
        else:
            # 添加 get_calc 实现
            def get_calc(model, board):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
                with torch.no_grad():
                    value, policy = model.calc(board_tensor)
                return float(value), policy.squeeze(0).cpu().numpy().tolist()

            org_value, org_policy = get_calc(self.model, root.board)
            good = 0
            for i in range(board_size):
                for j in range(board_size):
                    if root.board[i][j] == 0:
                        good += org_policy[i][j]
            for i in range(board_size):
                for j in range(board_size):
                    if root.board[i][j] != 0 or org_policy[i][j] == 0:
                        probs[i, j] = 0
                    else:
                        probs[i, j] = org_policy[i][j] / good

            sum_used = 0
            moves = []
            for move, (child, prior) in root.children.items():
                if child != None and child.visit_count != 0:
                    sum_used += probs[move]
                    probs[move] = 0
                    moves.append((move[0], move[1], child))
            moves.sort(key=lambda x: x[2].visit_count, reverse=True)
            value_sum = 0
            for i in range(len(moves)):
                cnt = moves[i][2].visit_count
                if i + 1 < len(moves):
                    cnt -= moves[i + 1][2].visit_count
                if cnt == 0:
                    continue
                cur = -1e9
                best_pos = i
                for j in range(i + 1):
                    vals = -moves[j][2].val
                    if vals > cur:
                        cur = vals
                        best_pos = j
                probs[moves[best_pos][0], moves[best_pos][1]] += sum_used * cnt * (i + 1) / total_visits
                value_sum += cur * cnt * (i + 1) / total_visits
            return value_sum, probs

    def get_train_data(self):
        boards, policies, values, weights = [], [], [], []
        for root in self.visited_nodes:
            child_count = sum((1 if child != None else 0) for child, prior in root.children.values())
            if not (child_count > 1 or root.visit_count >= Config.train_simulation / 2):
                continue

            # 添加 get_calc 实现
            def get_calc(model, board):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
                with torch.no_grad():
                    value, policy = model.calc(board_tensor)
                return float(value), policy.squeeze(0).cpu().numpy().tolist()

            probs = np.zeros((board_size, board_size))
            total_visits = sum((child.visit_count if child != None else 0) for child, prior in root.children.values())
            org_value, org_policy = get_calc(self.model, root.board)
            good = 0
            for i in range(board_size):
                for j in range(board_size):
                    if root.board[i][j] == 0:
                        good += org_policy[i][j]
            for i in range(board_size):
                for j in range(board_size):
                    if root.board[i][j] != 0 or org_policy[i][j] == 0:
                        probs[i, j] = 0
                    else:
                        org_policy[i][j] /= good
                        probs[i, j] = org_policy[i][j]

            sum_used = 0
            moves = []
            for move, (child, prior) in root.children.items():
                if child != None and child.visit_count != 0:
                    sum_used += probs[move]
                    moves.append((move[0], move[1], child, probs[move]))
                    probs[move] = 0
            moves.sort(key=lambda x: x[3], reverse=True)
            value_sum = 0
            for i in range(len(moves)):
                cnt = moves[i][3]
                if i + 1 < len(moves):
                    cnt -= moves[i + 1][3]
                if cnt == 0:
                    continue
                cur = -1e9
                best_pos = i
                for j in range(i + 1):
                    vals = -moves[j][2].val
                    if vals > cur:
                        cur = vals
                        best_pos = j
                probs[moves[best_pos][0], moves[best_pos][1]] += cnt * (i + 1)
                value_sum += cur * cnt * (i + 1) / sum_used
            boards.append(board_to_tensor(copy.deepcopy(root.board)))
            policies.append(torch.FloatTensor(probs))
            values.append(value_sum)
            weights.append(math.sqrt(total_visits / Config.train_simulation) * Config.train_buff)
        return boards, policies, values, weights