import torch
from torch import nn

from d2l import torch as d2l

batch_size = 256
hide_num = 512
hide_num2 = 64

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=64)

# net = nn.Sequential(nn.Flatten(),nn.Linear(4096,hide_num),nn.ReLU(),nn.Linear(hide_num,10))
net = nn.Sequential(nn.Flatten(),nn.Linear(4096,hide_num),nn.ReLU(),nn.Linear(hide_num,hide_num2),nn.ReLU(),nn.Linear(hide_num2,10))

def init_weights(m):
    print(type(m))
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 30
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()