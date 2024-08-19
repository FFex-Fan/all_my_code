# coding=utf-8
import math

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

df = pd.read_excel('../data/data1.xlsx')
df.columns = ['number', 'after30']

X = df['number'][1:].to_numpy().reshape(-1, 1)
y = df['after30'][1:].to_numpy().reshape(-1, 1)

# 生成多项式特征
degree = 2  # 多项式的度数，你可以更改这个值以尝试不同的多项式度数
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)  # 将原始特征转换为多项式特征

model = LinearRegression()
model.fit(X_poly, y)

X_test = np.linspace(0, 260, 260).reshape(-1, 1)
X_test_poly = poly.fit_transform(X_test)  # 将测试数据转化为多项式特征
y_test = model.predict(X_test_poly)

print(model.coef_)
print(model.intercept_)

plt.scatter(X, y, color='blue', label='people - C')
plt.plot(X_test, y_test, color='red', label='Fitting')
plt.xlabel('Number of people')
plt.ylabel('Real value')
plt.legend()

plt.savefig('../img/p2.png')
plt.show()
test = np.array([255]).reshape(-1, 1)
test_poly = poly.fit_transform(test)
num_255 = model.predict(test_poly).item()
print("当有 255 人游泳时，0.5 小时后水中余氯浓度为", num_255)


# 输出拟合的多项式函数表达式
def polynomial_expression(model, poly):
    terms = []
    intercept = model.intercept_[0]  # 将截距转换为标量
    terms.append(f"{intercept:.6f}")

    for i in range(1, model.coef_.size):  # 从1开始以跳过截距
        coef = model.coef_[0, i]
        term = poly.get_feature_names_out()[i]
        terms.append(f"{coef:+.6f} * {term}")

    return " ".join(terms)


expression = polynomial_expression(model, poly)
print("Fitted polynomial expression:")
print("y =", expression)

C_0 = 0.6
# 0.2972 = 0.6 * exp(-0.5 * k)
# -0.5 * k = ln(0.2972 / 0.6)
# k = -2 * ln(0.2972 / 0.6)
print('k =', -2 * math.log(0.2972 / 0.6, math.e))
# y = 0.6 * exp(-1.405 * t)
# 0.3 = 0.6 * exp(-1.405 * t)
# ln(0.5) = -1.405 * t
# t = ln(0.5) / -1.405

print("t =", math.log(0.5, math.e) / -1.405)
print("下一次加氯时间为：22:", (60 * 0.493).__ceil__())

#  y = 0.481254 -0.000312 * x0 -0.000002 * x0^2
