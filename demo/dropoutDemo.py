import torch
import torch.nn as nn

# 创建Dropout层
dropout = nn.Dropout(p=0.3)

# 输入数据
x = torch.ones(2, 5)
print("原始输入:")
print(x)

# 训练时的前向传播
x_drop = dropout(x)
print("\n训练时Dropout后:")
print(x_drop)

# 切换为评估模式
dropout.eval()
x_eval = dropout(x)
print("\n评估时(无Dropout):")
print(x_eval)