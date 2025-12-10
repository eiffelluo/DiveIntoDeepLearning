import torch
from torch import nn

# 定义一个延后初始化的网络
net = nn.Sequential(
    nn.LazyLinear(6),
    nn.ReLU(),
    nn.LazyLinear(4)
)

# 此时参数尚未初始化
print(net[0].weight)  # 输出：<UninitializedParameter>

# 输入数据触发初始化
X = torch.rand(2, 20)
net(X)

# 参数已初始化
print(net[0].weight.data.shape)  # 输出：[6,20]

print(net[2].weight.data.shape) # [4,6]

print(net.state_dict())


net = nn.Sequential(
    nn.Linear(20,3),
    nn.ReLU(),
    nn.Linear(3,2)
)

# print("--------------------------------")
# print(net[0].weight.data.shape)
# print(net[0].weight.data)