import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
res = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# print(res)

net = nn.Sequential(nn.Linear(8, 2), CenteredLayer())

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
        
net.apply(init_weights)

# X=torch.ones(4, 8)
# print(X)
# Y = net(X)
# print(Y)
# print(Y.mean())

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
linear = MyLinear(5, 3)
print(linear.weight)
print(linear.bias)
print(linear(torch.ones(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))