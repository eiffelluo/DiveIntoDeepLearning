import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data

NUM_WORKERS = 4
batch_size = 256

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def load_data_mnist(batch_size, resize=None):

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True)
    print('mnist_train length ',len(mnist_train))
    print('mnist_test length ',len(mnist_test))
    print('mnist_train[0] shape ',mnist_train[0][0].shape)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=NUM_WORKERS),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=NUM_WORKERS))
    
train_iter, test_iter = load_data_mnist(batch_size=batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()