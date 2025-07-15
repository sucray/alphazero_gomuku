# Gomoku AI (五子棋人工智能训练项目)

基于蒙特卡洛树搜索 (MCTS) 和深度强化学习的五子棋AI训练框架

## 项目结构
├── config.py # 配置文件 (超参数/路径设置) ├── main.py # 主训练循环 ├── train.py # 模型训练逻辑 ├── self_play.py # 自我对弈数据生成 ├── mcts.py # 蒙特卡洛树搜索实现 ├── data_utils.py # 数据预处理/增强工具 ├── model.py # 神经网络模型 (ValueCNN) └── README.md # 项目说明


## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8 (如需GPU加速)
- 其他依赖: `numpy`, `tqdm`, `matplotlib`

```bash
pip install torch numpy tqdm matplotlib
训练模型
python main.py
关键配置 (
config.py
)
# 棋盘/训练参数
board_size = 15                  # 棋盘大小 (15x15)
batch_size = 256                 # 训练批大小
num_epochs = 3                   # 每轮训练epoch数
learning_rate = 1e-4             # 学习率

# MCTS参数
train_simulation = 20            # 每次移动的模拟次数
dirichlet_alpha = 0.03           # 探索噪声参数
功能特性
深度强化学习

残差卷积网络 (Residual CNN)
双头输出 (策略+价值评估)
蒙特卡洛树搜索

带Dirichlet噪声的探索
并行化自我对弈
训练优化

数据增强 (旋转/翻转)
动态学习率调整
权重保存机制
自定义训练
修改
config.py
中的参数:

调整num_samples控制每轮生成的对局数
修改channel和num_blocks改变模型容量
使用预训练模型:

# 在config.py中设置
base_path = 'your_model.pth'
常见问题
错误处理
CUDA内存不足:

减小batch_size或num_samples
在
config.py
中启用fp16训练
设备不匹配错误:

# 确保模型和数据在同一设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
后续开发计划
[ ] 添加人机对战接口
[ ] 实现分布式训练
[ ] 集成TensorBoard可视化
许可证
MIT License













