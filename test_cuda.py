import torch
import numpy as np
from policy_value_net_pytorch import PolicyValueNet
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA不可用！"
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
# 初始化网络
net = PolicyValueNet(15, 15)
dummy_input = torch.randn(5, 4, 15, 15).cuda()

# 检查输出设备
log_probs, value = net.policy_value_net(dummy_input)
print(f"Log probabilities device: {log_probs.device}")  # 应输出: cuda:0
print(f"Value device: {value.device}")                  # 应输出: cuda:0



# 加载检查点（必须使用与保存时相同的环境）
print(f"验证环境: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

print(f"验证环境: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

data = torch.load(
    './checkpoints/checkpoint_10.checkpoint',
    map_location='cuda',
    weights_only=False
)

print("\n检查点内容:")
print(f"模型参数: {type(data['model_state']['conv1.weight'])} (device: {data['model_state']['conv1.weight'].device})")
print(f"优化器状态: {len(data['optimizer_state']['state'])}个参数组")
print(f"数据缓冲区: {len(data['data_buffer'])}条数据")
print(f"首条数据形状: {data['data_buffer'][0][0].shape}, {data['data_buffer'][0][1].shape}")