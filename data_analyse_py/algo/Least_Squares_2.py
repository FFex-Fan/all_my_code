# coding=utf-8

""" 二次多项式拟合 """
import numpy as np
import matplotlib.pyplot as plt

# 示例数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 1, 2, 2, 5, 10])

# 二次多项式拟合
X = np.vstack([x ** 2, x, np.ones(len(x))]).T
a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]

# 绘制数据点和拟合曲线
plt.scatter(x, y, label='Data Points')
plt.plot(x, a * x ** 2 + b * x + c, 'r', label=f'Fitted Curve: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
