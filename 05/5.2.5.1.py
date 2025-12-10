import torch
from torch import nn

X = torch.rand(size=(2, 3))


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(),
                                 nn.Linear(4, 2), nn.ReLU())
        self.linear = nn.Linear(2, 1)

    def forward(self, X):
        return self.linear(self.net(X))
    
net = NestMLP()
print(net(X))

print(net.state_dict())

print(net.net[0].weight)
print(net.net[0].bias)
print(net.net[2].weight)
print(net.net[2].bias)
print(net.linear.weight)
print(net.linear.bias)