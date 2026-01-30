import torch
import torch.nn as nn

# torch.set_printoptions(precision=8, sci_mode=False)

# 构造一个非常“好观察”的输入
X = torch.tensor([
    [[ 1.,  2.,  3.,  4.],
     [ 5.,  6.,  7.,  8.],
     [ 9., 10., 11., 12.]],

    [[13., 14., 15., 16.],
     [17., 18., 19., 20.],
     [21., 22., 23., 24.]]
])

# print("Input X shape:", X.shape)
# print(X)

ln_hidden = nn.LayerNorm(4, elementwise_affine=False)
Y_hidden = ln_hidden(X)

print("\nLayerNorm(4) 输出：")
print(Y_hidden)

# 验证：对每个 token 的 hidden 维度
print("\n每个 token 的 mean / var：")
print(Y_hidden.mean(dim=-1))
print(Y_hidden.var(dim=-1, unbiased=False))

ln_2d = nn.LayerNorm([3, 4], elementwise_affine=False)
Y_2d = ln_2d(X)

print("\nLayerNorm([3, 4]) 输出：")
print(Y_2d)

# 验证：对每个样本的 (T,H)
print("\n每个样本整体 mean / var：")
print(Y_2d.mean(dim=(1, 2)))
print(Y_2d.var(dim=(1, 2), unbiased=False))

