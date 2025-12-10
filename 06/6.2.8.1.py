import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X =torch.tensor([
    [1,1,1,0,0],
    [1,1,0,0,0],
    [1,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
])

# X = torch.eye(5)

print(X)
# print(X.t())

K= torch.tensor([
    [1.0,-1.0],
])



print(corr2d(X, K))
# print(corr2d(X.t(), K))
print(corr2d(X, K.t()))