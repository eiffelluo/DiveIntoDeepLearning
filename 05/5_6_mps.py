import torch
from torch import nn

print(torch.backends.mps.is_available())   # True 表示可以用 MPS
print(torch.backends.mps.is_built())       # True 表示 PyTorch 编译时支持 MPS

X = torch.ones(2, 3, device='mps')   # 存储在 MPS 上
print(X.device)   # 输出 mps

net = torch.nn.Sequential(torch.nn.Linear(3, 1))
net = net.to('mps')

Y = net(X)   # 输入在 MPS 上，计算也在 MPS 上
print(Y.device)   # 输出 mps
