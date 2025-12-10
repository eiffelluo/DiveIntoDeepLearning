import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(3, requires_grad=True)
print(a)
d = f(a)
print(d)
# 将张量输出转换为标量，以便进行反向传播
d.sum().backward()  # 或者使用 d.mean().backward()
print(a.grad)
