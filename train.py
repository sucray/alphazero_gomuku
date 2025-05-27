# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import time
import os
import torch
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
#from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


class TrainPipeline():
    def __init__(self, init_model=None, checkpoint_dir='./checkpoints'):
        # params of the board and the game
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 800  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 20000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 5000
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)  # 创建检查点目录
        self.start_batch = 0  # 记录当前训练轮次
        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=init_model
        )
        # 确保MCTS Player使用GPU
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )
        # 加载检查点（如果存在）
        if init_model is None:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                self._load_checkpoint(latest_checkpoint)

    def _find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.checkpoint')]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, latest)

    def _save_checkpoint(self, batch_idx):
        """优化保存检查点（避免Protocol 4警告）"""
        checkpoint = {
            'batch_idx': batch_idx,
            'model_state': {
                k: v.cpu().clone() for k, v in self.policy_value_net.get_policy_param().items()
            },
            'optimizer_state': {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.policy_value_net.optimizer.state_dict().items()
            },
            'data_buffer': [
                (x[0].copy(), x[1].copy(), float(x[2]))
                for x in self.data_buffer
            ],
            'best_win_ratio': float(self.best_win_ratio),
            'pure_mcts_playout_num': int(self.pure_mcts_playout_num)
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{batch_idx}.checkpoint')
        torch.save(checkpoint, path, pickle_protocol=2)  # 使用更兼容的协议
        print(f"检查点保存成功: {path}")

    def _load_checkpoint(self, path):
        """安全加载检查点（适配PyTorch 2.7+严格模式）"""
        import numpy as np
        from torch.serialization import add_safe_globals

        # 添加必要的全局变量到安全列表（使用numpy._core新路径）
        add_safe_globals([np._core.multiarray._reconstruct])  # 注意使用_numpy._core

        try:
            # 方案1：尝试weights_only模式
            checkpoint = torch.load(path, map_location='cuda', weights_only=True)
        except Exception as e:
            print(f"安全加载失败（{str(e)}），尝试非安全加载...")
            # 方案2：非安全加载（仅用于可信检查点）
            checkpoint = torch.load(path, map_location='cuda', weights_only=False)
        self.start_batch = checkpoint['batch_idx']
        # 恢复模型状态（确保转移到GPU）
        self.policy_value_net.policy_value_net.load_state_dict(
            {k: v.to('cuda') for k, v in checkpoint['model_state'].items()}
        )

        # 恢复优化器状态
        optimizer_state = {
            k: v.to('cuda') if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint['optimizer_state'].items()
        }
        self.policy_value_net.optimizer.load_state_dict(optimizer_state)

        # 恢复其他数据
        self.data_buffer = deque(checkpoint['data_buffer'], maxlen=self.buffer_size)
        self.best_win_ratio = checkpoint['best_win_ratio']
        self.pure_mcts_playout_num = checkpoint['pure_mcts_playout_num']
        print(f"从检查点恢复成功: 轮次={checkpoint['batch_idx'] + 1}")
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            print(f"开始第 {i} 次对局评测")
            ju_start = time.time()
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
            print(f"本次对对弈评测所花时间 {time.time() - ju_start:.2f}s")
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""

        try:
            for i in range(self.start_batch, self.game_batch_num):
                print("\n开始训练轮次：{} (总轮次: {}/{})".format(
                    i + 1,
                    i + 1,
                    self.game_batch_num
                ))
                total_start = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                print("轮次： i:{}, 本局总步数:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                # 每10局保存检查点
                if (i + 1) % 10 == 0:
                    self._save_checkpoint(i + 1)
                if (i+1) % self.check_freq == 0:
                    print("当前自对弈轮数: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("新的最佳策略产生!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                print(f"本轮训练所花时间 {time.time() - total_start:.2f}s")
        except KeyboardInterrupt:
            print("\n训练中断，正在保存最终检查点...")
            self._save_checkpoint(i + 1)  # 保存当前进度
            raise


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
