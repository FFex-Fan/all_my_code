# coding=utf-8
import numpy as np

# 生成数据
np.random.seed(0)
X = np.random.rand(100, 1) * 100  # 生成 100 个随机的特征值，范围为: (0, 10)
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # 生成目标函数，并加上噪声
# print(X, y)

# 添加偏置项
X_b = np.c_[np.ones((100, 1)), X]  # 在 X 前面加一列 1， 以表示偏置项（截距）

# 初始化权重
theta = np.random.randn(2)  # 随机初始化权重，包含截距和特征权重

# 超参数
eta = 0.01  # 学习率，控制每次更新的步长
epochs = 1000  # 迭代次数
lambda_val = 0.1  # 正则化参数，控制权重衰减的强度


# 定义损失函数（包含权重衰减项）
def calc_loss(X, y, theta, lambda_val):
    m = len(y)  # 样本数量
    predictions = X.dot(theta)  # 计算预测值
    mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # 计算均方误差
    regularization = (lambda_val / (2 * m)) * np.sum(theta[1:] ** 2)  # 计算权重衰减项（不包括截距）
    return mse + regularization


# 梯度下降算法
def grad_descent(X, y, theta, eta, epochs, lambda_val):
    m = len(y)  # 样本数量
    for epoch in range(epochs):
        grad = (1 / m) * X.T.dot(X.dot(theta) - y)  # 计算损失函数的梯度
        grad[1:] += (lambda_val / m) * theta[1:]  # 对权重进行正则化
        theta -= eta * grad  # 更新权重
    return theta  # 返回训练后的权重


# 训练函数
theta_optimal = grad_descent(X, y, theta, eta, epochs, lambda_val)

# 输出结果
print(f'训练后的权重：{theta_optimal}')
print(f'损失值：{calc_loss(X_b, y, theta_optimal, lambda_val)}')
