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



# X.shape (batch_size,num_step,dim)
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
        dropout_attention_w = self.dropout(attention_w)
        attention = torch.bmm(dropout_attention_w,value)
        self.attention_weights = dropout_attention_w
        return attention
      

# 原始/教学版 多头注意力 每个 head 真的当成一个独立的小注意力
class MulHeadAttention(nn.Module):
    def __init__(self,num_heads,d_model,query_size, key_size,value_size,num_hiddens,**kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_hiddens = num_hiddens
        self.qLiners = []
        self.kLiners = []
        self.vLiners = []
        # for i in range(num_heads):
        #     qLiner = nn.Linear(query_size, self.d_head)
        #     kLiner = nn.Linear(key_size, self.d_head)
        #     vLiner = nn.Linear(value_size, self.d_head)
        #     self.qLiners.append(qLiner)
        #     self.kLiners.append(kLiner)
        #     self.vLiners.append(vLiner)


        self.qLiners = nn.ModuleList([
            nn.Linear(query_size, self.d_head) for _ in range(num_heads)
        ])
        self.kLiners = nn.ModuleList([
            nn.Linear(key_size, self.d_head) for _ in range(num_heads)
        ])
        self.vLiners = nn.ModuleList([
            nn.Linear(value_size, self.d_head) for _ in range(num_heads)
        ])

        self.liner = nn.Linear(d_model,num_hiddens)
        self.attention = DotProductAttention(dropout=0)

    # queries的形状：(batch_size，查询的个数，d)
    def forward(self,query,key,value,valid_lens=None):
        head_outputs = []
        head_attentions = []
        for i in range(self.num_heads):
            q = self.qLiners[i](query)
            k = self.kLiners[i](key)
            v = self.vLiners[i](value)
            out_i = self.attention(q,k,v,valid_lens)
            attn_i = self.attention.attention_weights
            head_outputs.append(out_i)
            head_attentions.append(attn_i)

        concat = torch.concat(head_outputs,dim=-1)
        output = self.liner(concat)
        # print('output ----------')
        # print(output)
        # print('head_attentions --------')
        # print(head_attentions)
        return output

# 我的实现
# X 输入shape (batch_size，查询或者“键－值”对的个数，num_hiddens)
# 输出 shape (batch_size * num_heads ,查询或者“键－值”对的个数, num_hiddens // num_heads  )
def transpose_qkv(X, num_heads):
    shape = X.shape
    # 此时X shape (batch_size  ,查询或者“键－值”对的个数, num_heads, num_hiddens // num_heads  )
    X = X.reshape(shape[0],shape[1],-1,shape[2] // num_heads )
    # 此时X shape (batch_size  ,num_heads,查询或者“键－值”对的个数, num_hiddens // num_heads  )
    X = torch.transpose(X,1,2)
    # 此时 X shape (batch_size * num_heads ,查询或者“键－值”对的个数, num_hiddens // num_heads  )
    X = X.reshape(-1,shape[1],shape[2] // num_heads)
    return X

# 书中实现
def transpose_qkv2(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 输入 X shape (batch_size * num_heads ,查询或者“键－值”对的个数, num_hiddens // num_heads  )
# 输出 shape (batch_size，查询或者“键－值”对的个数，num_hiddens)
def transpose_output(X, num_heads):
    # 此时 X shape (batch_size , num_heads ,查询或者“键－值”对的个数, num_hiddens // num_heads  )
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    # 此时 X shape (batch_size ,查询或者“键－值”对的个数,num_heads , num_hiddens // num_heads  )
    X = torch.transpose(X,1,2)
    X = X.reshape(X.shape[0],X.shape[1],-1)
    return X


# 工程优化后的多头注意力   
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self,  query_size, key_size,value_size, num_hiddens,
                 num_heads, dropout=0, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
        self.attention = DotProductAttention(dropout)

     # queries，keys，values的形状:# (batch_size，查询或者“键－值”对的个数，num_hiddens)
    def forward(self,query,key,value,valid_lens=None):
        q = transpose_qkv(self.W_q(query),self.num_heads)
        k = transpose_qkv(self.W_k(key),self.num_heads)
        v = transpose_qkv(self.W_v(value),self.num_heads)

        if(valid_lens != None):
            valid_lens = torch.repeat_interleave(valid_lens,self.num_heads,dim=0)
        head_attention  = self.attention(q,k,v,valid_lens)
        # head_attention_weight = self.attention.attention_weights
        # print('head_attention_weight -------')
        # print(head_attention_weight)

        attention = transpose_output(head_attention,self.num_heads)
        # print('attention ------------')
        # print(attention)
        # attention_weight = transpose_output(head_attention_weight,self.num_heads)
        # print('attention_weight ------')
        # print(attention_weight)
        output = self.W_o(attention)
        # print('output  ---------')
        # print(output)
        return output



    



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
    

def testMulHeadAttention():
    X = torch.randn(2, 3, 4)
    mh = MulHeadAttention(num_head = 2,q_dim=4,k_dim=4,v_dim =4,num_hiddens=2)
    output,head_attentions = mh(X,X,X)
    print(output)
    print('-------------')
    print(head_attentions)

def testMulHeadAttention2():
    q = torch.randn(2, 3, 4)
    k = torch.randn(2,5,3)
    v = torch.randn(2,5,8)
    # mh = MulHeadAttention(num_heads = 2,query_size=4,key_size=3,value_size =8,num_hiddens=2,d_model=4)
    mh = MultiHeadAttention(num_heads = 2,query_size=4,key_size=3,value_size =8,num_hiddens=2)
    output,attention_weight = mh(q,k,v,valid_lens=torch.tensor([[1, 2,3], [1,2, 4]]))
   

def testTranspose_qkv():
    X = torch.arange(24).reshape(2,3,4)
    print(X)
    res = transpose_qkv(X,2)
    print(res.shape)
    print(res)
    Y = transpose_output(res,2)
    print(Y)

    # res2 = transpose_qkv2(X,2)
    # print(res2.shape)
    # print(res2)



# main()
# test_dot_product_attention()
# testMulHeadAttention2()
# testTranspose_qkv()