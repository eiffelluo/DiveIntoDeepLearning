import torch
import torch.nn.functional as F
from torch import nn

class TensorReduction(nn.Module):
    def __init__(self, dim1, dim2):
        super(TensorReduction, self).__init__()
        # 定义一个可训练的权重参数，维度为(dim2, dim1, dim1)
        self.weight = nn.Parameter(torch.ones(dim2, dim1, dim1))

    def forward(self, X):
        dim2 = self.weight.shape[0]
        Y = torch.zeros(X.shape[0], dim2)
        for k in range(dim2):
            t = (X @ self.weight[k]) @ X.T
            Y[:, k] = t.diagonal()
        return Y

# 创建一个TensorReduction层，dim1=10, dim2=5
layer = TensorReduction(2, 1)
# 创建一个大小为(2, 10)的张量X
X = torch.tensor([[1,2]],dtype=torch.float32)
print(X)
# 对layer(X)进行前向传播，返回一个大小为(2, 5)的张量
print(layer(X))
print(layer(X).shape)
