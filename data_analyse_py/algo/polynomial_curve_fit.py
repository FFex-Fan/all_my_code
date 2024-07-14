# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

""" 构造数据 """
x = np.linspace(-5, 5, 50)  # 在 (-5, 5) 等距生成 50 个点
y = np.sin(x) + np.random.rand(50)  # 生成y，y = sin(x) + 噪声

print(x.shape, y.shape)
plt.scatter(x, y)
plt.show()

""" 使用 numpy 进行多项式拟合 """
params = np.polyfit(x, y, 3) # 使用三次方多项式做拟合
print("params: ", params)

param_func = np.poly1d(params) # 构造一个便捷多项式对象（相当于直接给出拟合后的函数对象）
y_predict = param_func(x) # 根据原始 x，计算生成拟合后的 y_predict

plt.scatter(x, y)
plt.plot(x, y_predict, color='red')

plt.show()