import torch

# 设置打印选项，让张量显示更直观
torch.set_printoptions(precision=6, sci_mode=False)

feature_num = 5
lr = 0.1
num_epochs = 100
true_w = torch.tensor([0.5,0.1,-0.6,0.7,0.3]).reshape(-1,1)
train_num = 100
test_num = 50

def gen_features_labels(sample_num):
    features = torch.normal(0, 1, size=(sample_num, feature_num))
    labels = torch.mm(features,true_w)
    labels += torch.normal(0, 0.01, labels.shape)
    return features,labels


def init_weights(feature_num):
    return torch.normal(0, 0.01, size=(feature_num, 1), requires_grad=True)

def loss(y,y_hat):
    return ((y-y_hat).pow(2)).sum() / (2 * y.numel())

def net(X,w,b):
    return torch.mm(X,w) + b
    
def train(X,y,w,b):
    for i in range(num_epochs):
        print("epoch",i,"w",w)
        y_hat = net(X,w,b)
        l = loss(y,y_hat)
        print("epoch",i,"train loss",l)
            
        l.backward()
        # 使用 torch.no_grad() 更新参数，不破坏计算图
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            # 清零梯度
            w.grad.zero_()
            b.grad.zero_()
        
    return w,b

def predict(X,y,w,b):
    y_hat = net(X,w,b)
    l = loss(y,y_hat)
    print("predict loss",l)
    
    
init_w = init_weights(feature_num)
init_b = torch.zeros(1, requires_grad=True)
print('init_w',init_w,'init_b',init_b)

features,labels = gen_features_labels(train_num + test_num)

train_features = features[0: train_num]
train_labels = labels[0: train_num]
test_features = features[train_num:]
test_labels = labels[train_num:]
w,b = train(train_features,train_labels,init_w,init_b)
print('w',w,'b',b)
predict(test_features,test_labels,w,b)



