import torch
import torch.nn as nn

class PosEncoder(nn.Module):
    def __init__(self, dim,maxlen):
        super().__init__()
        self.d = dim
        self.maxlen = maxlen
        step = range(0,2,dim)
        print(step)


def posencode(dim,maxlen):
    P = torch.zeros(maxlen,dim)
    X = torch.arange(maxlen, dtype=torch.float32).reshape(-1, 1)
    
    #10000**(2j/dim)
    fenmu = torch.pow(10000,torch.arange(0, dim, 2, dtype=torch.float32)/dim)
    sin_X = X / fenmu
    # 如果dim为奇数，则cos_X 会比sin_X 少最后一个维度
    cos_X = sin_X if dim % 2 == 0 else sin_X[:,0:-1]
    P[:,0::2] = torch.sin(sin_X)
    P[:,1::2] = torch.cos(cos_X)
   




posencode(9,10)
# posencode(10,10)

# posencode2(9,10)