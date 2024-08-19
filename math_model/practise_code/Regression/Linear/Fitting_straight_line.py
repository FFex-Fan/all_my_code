# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

# 生成一些样本数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 误差，评估
train_MSE = mean_squared_error(y_train, y_train_pred)
test_MSE = mean_squared_error(y_test, y_test_pred)
score = model.score(X_test, y_test)
print('Train MSE:', train_MSE)
print('Test MSE:', test_MSE)
print('Test score:', score)

# 绘制结果
plt.scatter(X, y, color='blue', label='数据点')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(prop=my_font)
plt.title('线性回归拟合直线', fontproperties=my_font)
plt.show()

# 打印模型的系数和截距
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

"""
线性回归：
    形式：y = a0 + (a1 * x_1) + (a2 * x_2) + ··· + (an * x_n) + e
"""