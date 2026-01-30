import torch

# 简单数据
queries = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
keys = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])  # (1, 3, 2)

print("=" * 50)
print("步骤1: 原始数据")
print("=" * 50)
print(f"queries: {queries.shape}\n{queries}\n")
print(f"keys: {keys.shape}\n{keys}\n")

# unsqueeze
queries_exp = queries.unsqueeze(2)  # (1, 2, 1, 2)
keys_exp = keys.unsqueeze(1)        # (1, 1, 3, 2)

print("=" * 50)
print("步骤2: unsqueeze 扩展维度")
print("=" * 50)
print(f"queries_exp: {queries_exp.shape}\n{queries_exp}\n")
print(f"keys_exp: {keys_exp.shape}\n{keys_exp}\n")

# 广播相加
features = queries_exp + keys_exp

print("=" * 50)
print("步骤3: 广播相加（笛卡尔积）")
print("=" * 50)
print(f"features: {features.shape}\n{features}\n")

print("=" * 50)
print("步骤4: 验证笛卡尔积")
print("=" * 50)
for i in range(2):  # 2个查询
    for j in range(3):  # 3个键
        q = queries[0, i, :]
        k = keys[0, j, :]
        result = features[0, i, j, :]
        print(f"查询{i+1} {q} + 键{j+1} {k} = {result}")