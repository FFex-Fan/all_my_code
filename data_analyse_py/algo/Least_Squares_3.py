# coding=utf-8

""" 指数函数拟合 """
import numpy as np
import matplotlib.pyplot as plt


# 示例数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 2.7, 7.4, 20.1, 54.6, 148.4])

# 对 y 取对数
log_y = np.log(y)

# 线性拟合 log(y) = log(a) + bx
X = np.vstack([x, np.ones(len(x))]).T
b, log_a = np.linalg.lstsq(X, log_y, rcond=None)[0]
a = np.exp(log_a)

# 绘制数据点和拟合曲线
plt.scatter(x, y, label='Data Points')
plt.plot(x, a * np.exp(b * x), 'r', label=f'Fitted Curve: y = {a:.2f}e^({b:.2f}x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
