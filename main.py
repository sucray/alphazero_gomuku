# main.py
# 主训练循环
import torch
import os
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader
from config import Config
from model import ValueCNN
from self_play import generate_selfplay_data
from data_utils import Weighted_Dataset
from train import train_model

def work():
    """训练主流程"""
    config = Config()
    # 初始化模型
    model = ValueCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 添加设备检测
    model.to(device)  # 将模型移到设备（GPU/CPU）

    # 加载预训练模型，训练中断后可通过修改Config.py中base_path变量来加载训练模型以继续训练。
    if config.base_path != None:
        model.load_state_dict(torch.load(config.base_path, weights_only=True))
        model.to(device)

    # 训练循环
    for t in range(config.start_epo, config.end_epo):
        print('开始训练，当前轮次：',t+1)
        # 生成自我对弈数据
        model.eval()
        boards, policies, values, weights = generate_selfplay_data(model, config.num_samples)

        # 划分训练/验证集
        num_train = int(len(boards) * config.train_ratio)
        train_boards = boards[:num_train]
        train_policies = policies[:num_train]
        train_values = values[:num_train]
        train_weights = weights[:num_train]

        val_boards = boards[num_train:]
        val_policies = policies[num_train:]
        val_values = values[num_train:]
        val_weights = weights[num_train:]

        train_dataset = Weighted_Dataset(train_boards, train_policies, train_values, train_weights)
        val_dataset = Weighted_Dataset(val_boards, val_policies, val_values, val_weights)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # 训练模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, config)

        # 定期保存模型(每10轮保存一次)
        if t % 10 == 0:
            os.makedirs(config.model_path, exist_ok=True)
            checkpoint = os.path.join(config.model_path, f"{t + 1}.pth")
            torch.save(model.state_dict(), checkpoint)
            print(f"模型已保存为 {checkpoint}")

    return model

if __name__ == "__main__":
    # 初始化多进程环境
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn', force=True)
    # 启动训练
    work()