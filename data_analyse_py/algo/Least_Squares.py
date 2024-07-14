# coding=utf-8

# 最小二乘法

import numpy as np
import matplotlib.pyplot as plt

# 根据 y = 2 + 3x + 4x^2 生成数据点
X = np.arange(0, 5, 0.1)
Z = [2 + 3 * x + 4 * x ** 2 for x in X]
Y = np.array([np.random.normal(z, 3) for z in Z])

plt.plot(X, Y, 'bo')
plt.show()


# 生成系数矩阵A
def create_matrix(X, Y):
    n, m = len(X), 3  # n, m 分别表示 等式的个数，最高为2次方
    A = []

    for i in range(m):
        a = []
        # 计算当前方程中的每一个系数
        for j in range(m):
            a.append(sum(X ** (i + j)))
        A.append(a)

    return A


# 计算方程组右边的向量b
def get_right_vector(X, Y):
    n = len(X)
    m = 3
    B = []

    for i in range(m):
        B.append(sum(X ** i * Y))

    return B


A = create_matrix(X, Y)
b = get_right_vector(X, Y)

a0, a1, a2 = np.linalg.solve(A, b)

# 生成拟合曲线的绘制点
_X = np.arange(0, 5, 0.1)
_Y = np.array([a0 + a1 * x + a2 * x ** 2 for x in _X])

plt.plot(X, Y, 'bo', _X, _Y, 'r', linewidth=2)
plt.title("y = {} + {}x + {}$X^2$ ".format(a0, a1, a2))
plt.show()
