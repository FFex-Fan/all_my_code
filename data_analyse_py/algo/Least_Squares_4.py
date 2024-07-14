# coding=utf-8

""" 对数函数拟合 """
import numpy as np
import matplotlib.pyplot as plt

# 示例数据点
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 1, 1.6, 2, 2.3, 2.5])

# 对 x 取对数
log_x = np.log(x)

# 线性拟合 y = a * log(x) + b
X = np.vstack([log_x, np.ones(len(x))]).T
a, b = np.linalg.lstsq(X, y, rcond=None)[0]

# 绘制数据点和拟合曲线
plt.scatter(x, y, label='Data Points')
plt.plot(x, a * np.log(x) + b, 'r', label=f'Fitted Curve: y = {a:.2f}ln(x) + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
