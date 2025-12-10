import torch
import torch.nn.functional as F
from torch import nn


temp = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(temp)
print(temp.diagonal())
X = torch.ones(3,2)
X[:, 0] = temp.diagonal()
print(X)