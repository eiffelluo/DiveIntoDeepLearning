
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 记录程序开始时间
start_time = time.time()

batch_size, num_steps = 32, 35
time_machine = d2l.TimeMachine(batch_size, num_steps)
vocab = time_machine.vocab
train_iter = time_machine.get_dataloader(train=True)
sum = 0
for X, Y in train_iter:
    sum += X.shape[0]
    print('X: ', X)
    print('Y: ', Y)
    print('X shape: ', X.shape, 'Y shape: ', Y.shape)

print('sum: ', sum)