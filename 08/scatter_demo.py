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

# 生成随机数据
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='red', alpha=0.6, edgecolors='black', s=50)

# 添加回归线
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "b--", alpha=0.8, label='回归线')

plt.title('散点图示例', fontsize=16)
plt.xlabel('X值', fontsize=12)
plt.ylabel('Y值', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()