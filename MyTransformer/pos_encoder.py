import torch
import torch.nn as nn

# 设置科学计数法阈值
# torch.set_printoptions(sci_mode=False)  # 禁用科学计数法

class PosEncoder(nn.Module):
    def __init__(self,num_hiddens, dropout=0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros(1,max_len,num_hiddens)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        
        #10000**(2j/num_hiddens)
        fenmu = torch.pow(10000,torch.arange(0, num_hiddens, 2, dtype=torch.float32)/num_hiddens)
        sin_X = X / fenmu
        # 如果num_hiddens为奇数，则cos_X 会比sin_X 少最后一个维度
        cos_X = sin_X if num_hiddens % 2 == 0 else sin_X[:,0:-1]
        self.P[:,:,0::2] = torch.sin(sin_X)
        self.P[:,:,1::2] = torch.cos(cos_X)


    # X.shape (batch_size,num_step,dim)
    def forward(self,X):
        return self.dropout(self.P[:,:X.shape[1],:].to(X.device) + X)
    
def main(): 
    posEncoder = PosEncoder(8,10)
    X= torch.arange(160).reshape(2,10,8)
    pos = posEncoder(X)
    print(pos)
