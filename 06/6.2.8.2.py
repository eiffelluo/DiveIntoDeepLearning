import torch
from torch import nn
from d2l import torch as d2l

def conv2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias
    
    
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = conv2d(X, K)

    
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
# conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
c = Conv2D( kernel_size=(1, 2))

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1

lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = c(X)
    l = (Y_hat - Y) ** 2
    c.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    c.weight.data[:] -= lr * c.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(c.weight.data)
print(c.bias.data)