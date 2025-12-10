import torch
from torch import nn
from torch.nn import functional as F

X = torch.rand(2, 20)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 3), nn.ReLU(), nn.Linear(3, 1))
print(net(X))
d = net.state_dict()
print(d)
print(d['0.weight'])
print(d['0.bias'])
print(d['2.weight'])
print(d['2.bias'])


class MySequentialList(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.arr = []
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            # self._modules[str(idx)] = module
            self.arr.append(module)
            

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self.arr:
            X = block(X)
        return X
    
net = MySequentialList(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))
