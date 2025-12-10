
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):             # 定义 MLP 类
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)   # 定义隐藏层层，输入尺寸为 20，输出尺寸为 256
        self.output = nn.Linear(256, 10)   # 定义输出层，输入尺寸为 256，输出尺寸为 10

    def forward(self, x):          # 定义前向传播函数
        return self.output(F.relu(self.hidden(x)))  # 使用 ReLU 激活函数，计算隐藏层和输出层的输出

# 选择GPU，没有GPU就选CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建模型实例对象
net = MLP()
# 将模型参数传输到GPU上
net.to(device='mps')
# 访问模型参数
print(net.state_dict())
