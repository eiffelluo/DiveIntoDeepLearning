import torch
import torch.nn as nn

# 创建简单的输入数据: (batch_size=1, channels=1, height=3, width=3)
X = torch.tensor([[[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]]], dtype=torch.float32)

print("输入形状:", X.shape)  # torch.Size([1, 1, 3, 3])
print("输入数据:\n", X)

# 创建一个简单的卷积层: 1个输入通道, 1个输出通道, 2×2卷积核
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)

# 手动设置卷积核权重以便理解
# 卷积核形状: (out_channels, in_channels, kernel_height, kernel_width) = (1, 1, 2, 2)
conv.weight.data = torch.tensor([[[
    [1, 0],
    [0, 1]
]]], dtype=torch.float32)

print("\n卷积核形状:", conv.weight.shape)  # torch.Size([1, 1, 2, 2])
print("卷积核:\n", conv.weight.data)

# 执行卷积
output = conv(X)
print("\n输出形状:", output.shape)  # torch.Size([1, 1, 2, 2])
print("输出数据:\n", output)