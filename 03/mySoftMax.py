import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    
    
train_iter, test_iter = load_data_fashion_mnist(batch_size)


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)

# def cross_entropy(y,y_hat):
#     return -  (y* torch.log(y_hat)).sum(-1, keepdim=False)

# def cross_entropy(Y,Y_HAT):
#     return -(Y * torch.log(Y_HAT)).sum(-1,keepdim=False)

def cross_entropy(y,Y_HAT):
    # a = Y_HAT[:,y]
    a = Y_HAT[range(len(y)),y]
    return -torch.log(a)

def net(X,W,b):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W) + b)

def loss(Y,Y_HAT):
    return cross_entropy(Y,Y_HAT)

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
            
# 计算分类准确率
def accuracy(y_hat, y):
    with torch.no_grad():
        if len(y_hat.shape) > 1:
            y_hat = y_hat.argmax(dim=1)
        cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
        


W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
lr = 0.1
num_epochs = 10


for X, y in train_iter:
    print(X.shape, y.shape)
    y_hat = net(X,W,b)
    l = loss(y,y_hat)
    l.sum().backward()
    sgd([W, b], lr, batch_size)  # 使用参数的梯度更新参数
    
    with torch.no_grad():
        train_l = loss(y,net(X, W, b))
        # print(f'loss {float(train_l.mean()):f}')
        
# 重新创建训练数据加载器来读取所有训练样本
train_iter_all, _ = load_data_fashion_mnist(batch_size)

# 收集所有训练样本
all_X = []
all_y = []
for X, y in train_iter_all:
    all_X.append(X)
    all_y.append(y)

# 将所有batch的数据连接起来
X_all = torch.cat(all_X, dim=0)
y_all = torch.cat(all_y, dim=0)

print('------------')
print(f'所有训练样本形状: X={X_all.shape}, y={y_all.shape}')
train_acc = accuracy(net(X_all, W, b), y_all)
print(f'整体训练准确率: {train_acc}')

# 在最后统一打印预测和实际标签的信息
with torch.no_grad():
    y_hat_final = net(X_all, W, b)
    y_pred_classes = y_hat_final.argmax(dim=1)
    
    print(f'\n=== 预测和实际标签信息 ===')
    print(f'y_hat向量形状: {y_hat_final.shape}')
    print(f'y_hat向量类型: {y_hat_final.dtype}')
    print(f'预测类别形状: {y_pred_classes.shape}')
    print(f'预测类别类型: {y_pred_classes.dtype}')
    print(f'实际y形状: {y_all.shape}')
    print(f'实际y类型: {y_all.dtype}')
    print(f'前5个预测类别: {y_pred_classes[:100]}')
    print(f'前5个实际标签: {y_all[:100]}')
    print(f'预测准确率: {train_acc}/{len(y_all)} = {train_acc/len(y_all):.4f}')







