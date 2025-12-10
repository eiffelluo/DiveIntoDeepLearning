import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

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
# 特征数量
feature_num = 4096
# 类别
output_num = 10
lr = 0.01
num_epochs = 10
batch_size = 256

def softmax(O):
    # exp = torch.exp(O)
    Omax = O.max(dim=1,keepdim=True).values
    newO = O - Omax
    exp = torch.exp(newO)
    # s1 = exp.sum(dim=1).reshape(-1,1)
    # s1 = exp.sum(dim=1)
    s = exp.sum(dim=1,keepdim=True)
    result = exp / s
    return result

# def loss(y,Y_hat):
#     return -(y * torch.log(Y_hat[range(len(y)),y])).sum()

def loss(y,O):
    Omax = O.max(dim=1,keepdim=True).values
    newO = O - Omax
    exp = torch.exp(newO)
    s = exp.sum(dim=1,keepdim=True)
    logY_hat = newO - torch.log(s)
    return -(y *  logY_hat[range(len(y)),y] ).mean()

def net(X,W,b):
    return torch.mm(X,W) + b
    
train_iter, test_iter = load_data_fashion_mnist(1000, resize=64)
# for X, y in train_iter:
    
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     f = X[0].reshape(1,-1)
#     feature_num =f.shape[1]
#     print('feature_num ',feature_num)
#     break

def init_weights(feature_num,output_num):
    return torch.normal(0, 0.01, size=(feature_num, output_num), requires_grad=True)

def init_bias(output_num):
    return torch.zeros(output_num, requires_grad=True)

def train(W,b):
    
    for i in range(num_epochs):
        
        
        for X_source, y in train_iter:
            batch_num = len(X_source)
            X = X_source.reshape(batch_num,-1)
        
            O = net(X,W,b)
        
            l = loss(y,O)
            print('epoch ',i,'loss',l)
            # assert l < 0.5, l
            
            l.backward()
         
            
            with torch.no_grad():
                W -= W.grad * lr
                b -= b.grad * lr
                
                W.grad.zero_()
                b.grad.zero_()
                
def predict():
    correct = 0
    total = 0
    for X_source, y in test_iter:
        batch_num = len(X_source)
        X = X_source.reshape(batch_num,-1)
    
        O = net(X,W,b)
        Y_hat = softmax(O)
        y_pred = Y_hat.argmax(dim=1)
        # print('y_pred ',y_pred)
        # print('y ',y)
        print('accuracy ',(y_pred == y).sum() / len(y))
        correct += (y_pred == y).sum()
        total += len(y)
    print('correct ',correct)
    print('total ',total)
    print('total accuracy ',correct / total)

W = init_weights(feature_num,output_num)
b = init_bias(output_num)
train(W,b)
# print('W ',b)
# print('b ',b)
predict()







