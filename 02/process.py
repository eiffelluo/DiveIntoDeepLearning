import torch

def f(a):
    if a < 3:
        return 2*a
    else:
        return a*a
    
x = torch.tensor(2.0, requires_grad=True)
d = f(x)
d.backward()
print(x.grad)