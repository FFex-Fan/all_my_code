# coding=utf-8

""" 线性拟合 """
import numpy as np
import matplotlib.pyplot as plt

# 示例数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 7, 8])

# 添加一个常数项以表示截距项
X = np.vstack([x, np.ones(len(x))]).T

# 使用最小二乘法计算系数
a, b = np.linalg.lstsq(X, y, rcond=None)[0]

# 绘制数据点和拟合直线
plt.scatter(x, y, label='Data Points')
plt.plot(x, a*x + b, 'r', label=f'Fitted Line: y = {a:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
