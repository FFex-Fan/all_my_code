# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

""" 整体思路：
        设定目标函数 y = wx + b
        目标：根据给定的 (x, y) 坐标，以均方损失作为损失函数，使用梯度下降，找到 w, b 的近似值
"""

# 数据点 => y = 2x
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

# 超参数
alpha = 0.01  # 学习率
epochs = 40  # 迭代次数

# 初始化参数 (随机值)
b = np.random.randn()
w = np.random.randn()

# 记录损失
loss_history = []

# 梯度下降
m = len(x)
for epoch in range(epochs):
    y_pred = b + w * x
    d_b = (-2 / m) * np.sum(y - y_pred)
    d_w = (-2 / m) * np.sum((y - y_pred) * x)
    b -= alpha * d_b
    w -= alpha * d_w

    # 计算损失
    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

print(f'b: {b}, w: {w}')
print(f'Final loss: {loss_history[-1]}')

plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
