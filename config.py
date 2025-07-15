# config.py
# 五子棋AI训练配置文件

# 棋盘参数
board_size = 15  # 棋盘大小 (15x15 标准五子棋)


class Config:
    # 训练超参数
    batch_size = 256  # 每个训练批次的样本数
    num_epochs = 3  # 每轮自对弈数据训练的epoch数
    learning_rate = 1e-4  # 初始学习率
    train_ratio = 0.9  # 训练集/验证集分割比例

    # 数据生成参数
    num_samples = 100  # 每轮生成的自我对弈局数
    channel = 32  # 神经网络卷积通道数
    num_workers = 5  # 并行生成数据的进程数

    # MCTS参数
    train_simulation = 20  # 每次移动的MCTS模拟次数
    dirichlet_alpha = 0.03  # 探索噪声参数α
    dirichlet_epsilon = 0.25  # 探索噪声混合比例

    # 路径配置
    base_path = None  # 预训练模型路径
    model_path = 'gomoku_cnn'  # 模型保存目录

    # 训练控制
    start_epo = 0  # 起始训练轮次
    end_epo = 50000  # 结束训练轮次
    train_buff = 0.8  # 训练样本权重系数

    # 调试选项
    output_info = True  # 是否输出训练信息
    collect_subnode = True  # 是否收集MCTS子节点数据