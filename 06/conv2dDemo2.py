import torch
import torch.nn as nn

# 输入: 2个通道, 每个通道是 3×3
X = torch.tensor([
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ]
    ]
], dtype=torch.float32)

print("输入形状:", X.shape)  # torch.Size([1, 2, 3, 3])

# 2个输入通道, 1个输出通道, 2×2卷积核
conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2, bias=False)

# 卷积核形状: (1, 2, 2, 2) - 1个输出通道, 2个输入通道, 每个是2×2
# 第一个通道的卷积核: [[1, 0], [0, 1]]
# 第二个通道的卷积核: [[0, 1], [1, 0]]
conv.weight.data = torch.tensor([[[
    [1, 0],
    [0, 1]
], [
    [0, 1],
    [1, 0]
]]], dtype=torch.float32)

print("\n卷积核形状:", conv.weight.shape)  # torch.Size([1, 2, 2, 2])

output = conv(X)
print("\n输出形状:", output.shape)  # torch.Size([1, 1, 2, 2])
print("输出数据:\n", output)