import torch
import torch.nn as nn

# 设置科学计数法阈值
# torch.set_printoptions(sci_mode=False)  # 禁用科学计数法

class FeatureNorm(nn.Module):
    def __init__(self):
        super().__init__()

    # X.shape (batch_size, num_step , num_hiddens)
    def forward(self,X):
        mean = X.mean(dim=-1,keepdim=True)
        std = X.std(dim=-1,keepdim=True)
        X2 = (X - mean)/std
        return X2
    

class Add(nn.Module):
    def __init__(self,dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,X,Y):
        return X + self.dropout(Y)


def main():
    X = torch.arange(12,dtype=torch.float).reshape(3,4)
    fn = FeatureNorm()
    print(fn(X))

main()
