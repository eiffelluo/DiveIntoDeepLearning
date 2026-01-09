import torch
import torch.nn.functional as F

# 假设我们有一个概率分布
probs = torch.tensor([1.0, 2.0, 7.0])
probs = F.softmax(probs, dim=0)
print(probs)

# 从这个分布中采样 1 个元素
for i in range(10):
    sample = torch.multinomial(probs, num_samples=1)
    print(sample)  # 可能输出 tensor([2])，因为第三个元素概率最大
    # print(probs[sample])  # 可能输出 tensor([0.7])，因为第三个元素概率最大
