import torch
from torch import nn
import time

start = time.perf_counter()

n = 100
d = 'cpu'

X = torch.rand(n, n, device=d)  
Y = torch.rand(n, n, device=d)
Z = X @ Y @ X @ Y @ X @ Y @ X @ Y @ X @ Y

    

end = time.perf_counter()
elapsed = end - start
print(f"执行时间: {elapsed:.4f} 秒")