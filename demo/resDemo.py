import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

N = 1024        # 样本数
D = 128         # 维度

x = torch.randn(N, D)
y = x.clone()

class PlainMLP(nn.Module):
    def __init__(self, depth=10, dim=128):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.fc(x))


class ResMLP(nn.Module):
    def __init__(self, depth=10, dim=128):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlock(dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)

def train(model, x, y, steps=500, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:03d}, loss = {loss.item():.6f}")

plain = PlainMLP(depth=10, dim=D)
print("Training Plain MLP")
train(plain, x, y)

resnet = ResMLP(depth=10, dim=D)
print("\nTraining Residual MLP")
train(resnet, x, y)
