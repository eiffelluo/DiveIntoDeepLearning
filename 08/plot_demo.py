# 安装 Matplotlib
# pip install matplotlib

# 导入库
import matplotlib.pyplot as plt
import numpy as np
import platform

# 根据操作系统选择合适的中文字体
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'SimHei']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 创建图形和坐标轴
plt.figure(figsize=(8, 6))  # 设置图形大小

# 绘制折线图
plt.plot(x, y, marker='s', linestyle='-', color='r', linewidth=1, label='y=2x')

# 添加标题和标签
plt.title('简单的折线图示例', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)

# 添加图例
plt.legend()

# 添加网格
plt.grid(True, alpha=0.3)

# 显示图形
plt.show()