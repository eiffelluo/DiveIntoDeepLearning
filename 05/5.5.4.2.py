import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
    
net = MLP()

torch.save(net.hidden.state_dict(), 'mlp.hidden')

net2 = MLP()
net2.hidden.load_state_dict(torch.load('mlp.hidden'))

print(net2.hidden.weight)
print(net.hidden.weight)