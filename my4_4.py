import math
# import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

true_b = 5
max_f = 20
temp_true_w = torch.zeros(max_f)
temp_true_w[0:3] = torch.tensor([1.2,-3.4,5.6])
true_w = temp_true_w.reshape(-1,1)

def factorial_reciprocal(n):
    """计算 1 到 n 的阶乘的倒数"""
    k = torch.arange(1, n+1, dtype=torch.float32)  # [1, 2, 3, ..., n]
    factorial = torch.cumprod(k, dim=0)  # [1!, 2!, 3!, ..., n!]
    return 1.0 / factorial  # [1/1!, 1/2!, 1/3!, ..., 1/n!]


coefficients = factorial_reciprocal(max_f)

def gen_features(sample_num):
    X = torch.normal(0, 1, size=(sample_num,1))
    # X = torch.tensor([[2],[3]])
    # 创建幂次张量 [1, n]
    powers = torch.arange(1, max_f+1).unsqueeze(0)
    # print(powers)
    # 计算所有幂次 [m, n]
    power_matrix = X.pow(powers)
    # print(power_matrix)
  
    # 系数张量调整为 [1, n] 以支持广播
    coeffs = coefficients.unsqueeze(0)
    
    # 加权后的幂次矩阵 [m, n]
    features = power_matrix * coeffs
    # print(features)
    return features

def compute_labels(features):
    return torch.mm(features,true_w) + true_b
    
    
# features = gen_features()
# labels = compute_labels(features)
# print(labels)

def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=True))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
    print('bias:', net[0].bias.data.numpy())

def test(feature_num):
 
    train_X = gen_features(100)
    train_y = compute_labels(train_X)
    train_y += torch.normal(0, 0.1, train_y.shape)
    train_features = train_X[:,0: feature_num]
    
    test_X = gen_features(100)
    test_y = compute_labels(test_X)
    test_y += torch.normal(0, 0.1, test_y.shape)
    test_features = test_X[:,0: feature_num]
    
    # n_train, n_test = 100, 100  # 训练和测试数据集大小
    # X = gen_features(n_train + n_test)
    # y = compute_labels(X)
    train(train_features,test_features,train_y,test_y,400)
    d2l.plt.show()
     

test(2)