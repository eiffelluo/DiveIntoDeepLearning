import torch

A = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])  # shape (3, 3)

v = torch.tensor([2., 3., 4.])  # shape (3,)

# 关键：调整 v 的形状为 (m, 1)，使其能广播到 (m, n)
result = A / v.reshape(-1, 1)  # 或 v.unsqueeze(1)

print(result)