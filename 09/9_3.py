import torch
from torch import nn
from d2l import torch as d2l

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mydl

batch_size, num_steps = 32, 35
train_iter, vocab = mydl.load_data(batch_size, num_steps, data_source='time_machine')

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = mydl.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = mydl.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
mydl.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)

d2l.plt.show()