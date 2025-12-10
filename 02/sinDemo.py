import torch
import math

x = torch.linspace(-math.pi, math.pi, 4,requires_grad=True)
print(x)
# x = torch.tensor(math.pi,requires_grad=True)
y = torch.sin(x)
y.sum().backward()
print(x.grad)



