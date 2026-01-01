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

# 创建2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('多子图示例', fontsize=16)

# 生成示例数据
x = np.linspace(0, 10, 1000)

# 左上子图：正弦曲线
axes[0, 0].plot(x, np.sin(x), 'r-')
axes[0, 0].set_title('正弦函数')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# 右上子图：余弦曲线
axes[0, 1].plot(x, np.cos(x), 'g--')
axes[0, 1].set_title('余弦函数')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

# 左下子图：柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]
axes[1, 0].bar(categories, values, color='skyblue', edgecolor='black')
axes[1, 0].set_title('柱状图')
axes[1, 0].set_xlabel('类别')
axes[1, 0].set_ylabel('数值')

# 右下子图：饼图
sizes = [15, 30, 25, 20, 10]
labels = ['Python', 'Java', 'C++', 'JavaScript', '其他']
axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('编程语言占比')

# 调整布局
plt.tight_layout()
plt.show()