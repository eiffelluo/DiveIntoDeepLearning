import torch

# x = torch.tensor([1, 2, 3])
# print(x.shape)  # torch.Size([3])

# # 沿第0维重复2次
# y = x.repeat(2)
# print(y)        # tensor([1, 2, 3, 1, 2, 3])
# print(y.shape)  # torch.Size([6])

x = torch.tensor([[1, 2], [3, 4]])
print(x.shape)  # torch.Size([2, 2])

# 第0维重复2次，第1维重复3次
y = x.repeat(2, 3)
print(y)
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])
print(y.shape)  # torch.Size([4, 6])