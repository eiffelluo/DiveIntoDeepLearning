# 代码验证
import torch
import torch.nn as nn


class MyNet1(nn.Module):
    def __init__(self, linear1, linear2):
        super(MyNet1, self).__init__()
        self.linear1 = linear1
        self.linear2 = linear2

    def forward(self, X):
        return self.linear2(self.linear1(nn.Flatten()(X)))


class MyNet2(nn.Module):
    def __init__(self, linear, conv2d):
        super(MyNet2, self).__init__()
        self.linear = linear
        self.conv2d = conv2d

    def forward(self, X):
        X = self.linear(nn.Flatten()(X))
        X = X.reshape(X.shape[0], -1, 1, 1)
        print('X.shape',X.shape)
        X = nn.Flatten()(self.conv2d(X))
        return X


linear1 = nn.Linear(15, 10)
linear2 = nn.Linear(10, 5)
conv2d = nn.Conv2d(10, 5, 1)
print('linear2.weight',linear2.weight)
linear2.weight = nn.Parameter(conv2d.weight.reshape(linear2.weight.shape))  
print('linear2.weight',linear2.weight)
linear2.bias = nn.Parameter(conv2d.bias)

net1 = MyNet1(linear1, linear2)
net2 = MyNet2(linear1, conv2d)

X = torch.randn(2, 3, 5)
# 两个结果实际存在一定的误差，直接print(net1(X) == net2(X))得到的结果不全是True
print(net1(X))
print(net2(X))
