import torch
import torch.nn as nn
import math

MIN_NUM = -20

def masked_softmax(X, valid_lens):
    if(valid_lens == None):
        return torch.softmax(X,dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1 :
        tmp_valid_lens = torch.repeat_interleave(valid_lens,shape[1])
    if valid_lens.dim() == 2 :
        tmp_valid_lens = valid_lens.reshape(-1)
  
    print(X)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self):
        super(DotProductAttention, self).__init__()

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self,query,key,value,valid_lens=None):
        d = query.shape[-1]
        score = torch.bmm(query,key.transpose(1,2))/math.sqrt(d)
        attention_w = masked_softmax(score,valid_lens)
        attention = torch.bmm(attention_w,value)
        return attention
      

class MulHeadAttention(nn.Module):
    def __init__(self,num_head,q_dim,k_dim,v_dim,o_dim,dim):
        super().__init__()
        self.num_head = num_head
        self.o_dim = o_dim
        self.qLiner = nn.Linear(q_dim,o_dim)
        self.kLiner = nn.Linear(k_dim,o_dim)
        self.vLiner = nn.Linear(v_dim,o_dim)
        self.liner = nn.Linear(o_dim,dim)

    def forward(self,query,key,value):
        q = self.qLiner(query)
        k = self.kLiner(key)
        v = self.vLiner(value)
        score = torch.bmm(q,k.transpose(1,2))/math.sqrt(self.o_dim)
        attention_w = torch.softmax(score,dim=-1)
        attention = torch.bmm(attention_w,value)
        return attention



def main():
    mh = MulHeadAttention(1,3,3,1,2,2)
    query = torch.tensor([
        [
            [1,2,3],
            [2,3,4]
        ]
    ],dtype=torch.float)

    key = torch.tensor([
        [
            [1,0,1],
            [1,1,1],
        ]
    ],dtype=torch.float)

    value = torch.tensor(
        [
            [
                [10],
                [2]
            ]
        ],dtype=torch.float
    )

    attention = mh(query,key,value)
    print(attention)

main()