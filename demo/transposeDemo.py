import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
x.shape  # (2, 3)

y = x.transpose(1, 0)
y.shape  # (3, 2)
print(y)