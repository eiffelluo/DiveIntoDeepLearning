import torch
import torch.nn as nn

embedding = nn.Embedding(5, 3)  # 词汇表大小=5，每个词用3维向量表示
input = torch.LongTensor([0, 2, 4])  # 输入词索引
output = embedding(input)

print(output)

print(embedding.weight)
