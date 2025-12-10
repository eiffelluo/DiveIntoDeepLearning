import torch
from torch import nn
from torch.nn import functional as F

X = torch.ones(1, 3)

class MySequential(nn.Module):
    def __init__(self, net1,net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
      o1 = self.net1(X)
      o2 = self.net2(X)
      return torch.cat((o1,o2),dim=1)
        
    
    
net = MySequential(nn.Linear(3, 1), nn.Linear(3, 1))
print(net(X))
print(net.state_dict())