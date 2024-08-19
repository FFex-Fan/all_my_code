# coding=utf-8
import numpy as np
import sklearn.linear_model as lm

X = np.array([9, 13, 15, 17, 18, 26, 22, 20, 23, 28, 30, 33]).reshape(-1, 1)
y = np.array([3, 5, 4, 6, 7, 9, 8, 7, 10, 11, 10, 12]).reshape(-1, 1)

model = lm.LinearRegression()
model.fit(X, y)
print(f"y_hat = {model.intercept_.item()} + {model.coef_.item()}x")

y_pred = model.predict(X)
y_avg = np.full((12, 1), np.mean(y, axis=0))
SSE = np.sum((y_pred - y) ** 2, axis=0)
SSR = np.sum((y_pred - y_avg) ** 2, axis=0)
F = (SSR / 1) / (SSE / (X.shape[0] - 1 - 1))
print(f"SSE = {SSE.item()}")
print(f"SSR = {SSR.item()}")
print(f"F = {F.item()}")

sigma_squared = SSE / (12 - 1 - 1)
variance_X = np.sum((X - np.mean(X)) ** 2)
SE_beta_1 = np.sqrt(sigma_squared / variance_X)
numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
denominator = np.sum((X - np.mean(X)) ** 2)
beta_1 = numerator / denominator
print(f"SE_beta_1 = {SE_beta_1.item()}") # 越低说明越拟合
print(f"t_stat = {beta_1 / SE_beta_1}")