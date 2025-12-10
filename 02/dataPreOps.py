import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    
# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)

# 分别处理数值型和分类型特征
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]

# 对数值型特征（NumRooms）使用均值填充
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
# 对分类型特征（Alley）使用众数填充
inputs['Alley'] = inputs['Alley'].fillna(inputs['Alley'].mode()[0])

print(inputs)

import torch

# X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(y)