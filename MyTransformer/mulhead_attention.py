import torch
import torch.nn as nn
import math

torch.set_printoptions(sci_mode=False)  # 禁用科学计数法


# X.shape (num_size,dim)

def sequence_mask(X, valid_len, value=0):
    dim = X.size(1)
    new_X = torch.arange(dim,device=X.device)[None,:]
    new_valid_len = valid_len[:,None]
    mask = new_X < new_valid_len
    X[~mask] = value
    return X



# X.shape (batch_size,num_size,dim)
def masked_softmax(X, valid_lens):
    if(valid_lens == None):
        return torch.softmax(X,dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1 :
        tmp_valid_lens = torch.repeat_interleave(valid_lens,shape[1])
    if valid_lens.dim() == 2 :
        tmp_valid_lens = valid_lens.reshape(-1)
  
    X = sequence_mask(X.reshape(-1, shape[-1]), tmp_valid_lens,value=-1e6)
    return torch.softmax(X.reshape(shape),dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self,query,key,value,valid_lens=None):
        d = query.shape[-1]
        score = torch.bmm(query,key.transpose(1,2))/math.sqrt(d)
        attention_w = masked_softmax(score,valid_lens)
        print(attention_w)
        dropout_attention_w = self.dropout(attention_w)
        print(dropout_attention_w)
        attention = torch.bmm(dropout_attention_w,value)
        return attention
      

class MulHeadAttention(nn.Module):
    def __init__(self,num_head,q_dim,k_dim,v_dim,o_dim,dim):
        super().__init__()
        self.num_head = num_head
        self.o_dim = o_dim
        self.qLiners = []
        self.kLiners = []
        self.vLiners = []
        for i in range(num_head):
            qLiner = nn.Linear(q_dim,o_dim)
            kLiner = nn.Linear(k_dim,o_dim)
            vLiner = nn.Linear(v_dim,o_dim)
            self.qLiners.append(qLiner)
            self.kLiners.append(kLiner)
            self.vLiners.append(vLiner)
        self.liner = nn.Linear(o_dim,dim)
        self.attention = DotProductAttention(dropout=0.5)

    def forward(self,query,key,value):
        pass
      



def test_dot_product_attention():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))

def main():
    mh = MulHeadAttention(1,3,3,1,2,2)
    query = torch.tensor([
        [
            [1,0,1],
            [1,2,3]
        ]
    ],dtype=torch.float)

    key = torch.tensor([
        [
            [-1,0,1],
            [1,2,3],
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

    dpa = DotProductAttention(0.25)
    print(dpa(query,key,value))

    # attention = mh(query,key,value)
    # print(attention)
    # X = torch.rand(2, 2, 4)
    # print(X)
    # print(masked_softmax(X,torch.tensor([2, 3])))
    
    # print('----------------')

    # X = torch.rand(2, 2, 4)
    # print(X)
    # print(masked_softmax(X,torch.tensor([[1, 3], [2, 4]])))
    

# main()
test_dot_product_attention()