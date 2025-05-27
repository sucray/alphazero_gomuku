# AlphaZero Gomoku Implementation

基于 AlphaZero 算法的五子棋 (Gomoku) 实现，使用 PyTorch 和 GPU 加速。

## 环境要求
- Python 3.7+
- PyTorch 2.0+ (推荐 2.7.0+cu128)
- CUDA 11.8+ (如需 GPU 加速)
- numpy, collections, os, time 等基础库

## 文件结构
. ├── train.py # 主训练脚本 ├── game.py # 棋盘逻辑和游戏规则 ├── policy_value_net_pytorch.py # 策略价值网络 (PyTorch 实现) ├── mcts_pure.py # 纯蒙特卡洛树搜索实现 ├── mcts_alphaZero.py # AlphaZero 风格的 MCTS └── checkpoints/ # 训练检查点保存目录


## 快速开始

### 
1. 安装依赖
```bash
pip install torch numpy

2. 开始训练
python train.py
3. 恢复训练
python train.py --init_model ./checkpoints/checkpoint_10.checkpoint

关键功能
训练参数 (train.py)
| 参数 | 默认值 | 说明 | |------|--------|------| | n_playout | 800 | 每次移动的模拟次数 | | buffer_size | 20000 | 经验回放缓冲区大小 | | batch_size | 512 | 训练批次大小 | | check_freq | 50 | 模型评估频率 | | game_batch_num | 5000 | 总训练轮次 |

GPU 加速
自动检测 CUDA 并启用 GPU 加速
支持多卡训练 (需手动修改代码)
断点续训
每10局自动保存检查点到 ./checkpoints
支持键盘中断时自动保存
恢复训练自动从最新检查点继续
自定义配置
修改棋盘大小
在 
train.py
 中调整：
self.board_width = 15  # 棋盘宽度
self.board_height = 15 # 棋盘高度
self.n_in_row = 5      # 几连珠获胜

