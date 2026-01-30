import torch
from torch import nn
import math

# 假设你已经有 DotProductAttention
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if valid_lens is not None:
            mask = torch.arange(keys.size(1), device=queries.device)[None, :] >= valid_lens[:, None]
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, values)
        return output, attn_weights

# 保留你原来的 MultiHeadAttention，但 forward 返回 attn_weights
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output, attn_weights = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attn_weights

# ========== 测试多头 attention ==========
batch_size, num_queries, num_kvpairs, num_hiddens, num_heads = 2, 4, 6, 100, 5
valid_lens = torch.tensor([3, 2])

attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.0)

# 输入全是 ones，为了演示每个 head 的 attention 可能不同
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

output, attn_weights = attention(X, Y, Y, valid_lens)

print("Output shape:", output.shape)  # (batch_size, num_queries, num_hiddens)
print("Attention weights shape:", attn_weights.shape)  # (batch_size*num_heads, num_queries, num_kvpairs)

# 让我们看一下第一个 batch 的每个 head 的 attention map
for head in range(num_heads):
    print(f"\nHead {head+1} attention weights (first batch):")
    print(attn_weights[head].detach())
