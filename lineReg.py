import torch
import torch.nn as nn
import torch.optim as optim

# 设置打印选项，让张量显示更直观
torch.set_printoptions(precision=6, sci_mode=False)

feature_num = 5
out_features = 1
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

net = nn.Sequential(nn.Linear(feature_num,out_features,bias=True))

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

features,labels = gen_features_labels(train_num + test_num)

train_features = features[0: train_num]
train_labels = labels[0: train_num]
test_features = features[train_num:]
test_labels = labels[train_num:]

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

for epoch in range(num_epochs):
    trainer.zero_grad()
    l = loss(net(train_features) ,train_labels)
    print(f'epoch: {epoch} loss: {l:f}')
    l.backward()
    trainer.step()
    
print(f'w: {net[0].weight.data}')
print(f'b: {net[0].bias.data}')


l = loss(net(test_features) ,test_labels)
print(f'test loss: {l:f}')

    

    
    



