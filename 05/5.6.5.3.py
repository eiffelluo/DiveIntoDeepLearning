import torch
import time

# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
device = 'mps'

# 生成随机矩阵
matrices = [torch.randn(100, 100).to(device) for i in range(1000)]

for i in range(1000):
    result = torch.mm(matrices[i], matrices[i].t())
    frobenius_norm = torch.norm(result)

# 实验一：计算时间
start_time_1 = time.time()
for i in range(1000):
    result = torch.mm(matrices[i], matrices[i].t())
    frobenius_norm = torch.norm(result)
    # print(frobenius_norm)
end_time_1 = time.time()
print("Time taken:", end_time_1 - start_time_1)

# 实验二：计算时间
start_time_2 = time.time()
for i in range(1000):
    result = torch.mm(matrices[i], matrices[i].t())
    frobenius_norm = torch.norm(result)
    print(frobenius_norm)
end_time_2 = time.time()
print("Time taken:", end_time_2 - start_time_2)

print(f'实验一消耗时间：{end_time_1 - start_time_1}，实验二消耗时间：{end_time_2 - start_time_2}')
