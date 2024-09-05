# coding=utf-8
import numpy as np
import sympy as sp

if __name__ == '__main__':
    x0 = np.array([0, 71.1, 72.4, 72.4, 72.1, 71.4, 72.0, 71.6]).reshape(-1, 1)  # 列向量
    n = len(x0) - 1

    # 级比检验
    lambda_val = x0[1: n] / x0[2: n + 1]
    range_max, range_min = np.max(lambda_val), np.min(lambda_val)
    lambda_max, lambda_min = np.exp(2 / (n + 1)), np.exp(-2 / (n + 1))
    if range_min > lambda_min and range_max < lambda_max:
        print("原始数据通过级比检验")
    else:
        print("原始数据未能通过级比检验!!")
        exit(0)

    x1 = np.cumsum(x0).T  # 前缀和
    print("x1_sum: ", x1)
    tmp_matrix = x1[1: n] + x1[2: n + 1]
    B = np.array([-0.5 * tmp_matrix]).reshape(-1, 1)  # 将矩阵转化为 6 x 1 的形状
    B = np.hstack((B, np.ones((6, 1))))  # 水平方向拼接上 1 矩阵
    Y = np.array(x0[2: n + 1])
    print("B 矩阵：\n", B)
    print("Y 矩阵：\n", Y)
    u, residuals, rank, s = np.linalg.lstsq(B, Y, rcond=None)
    u = np.round(u, 5)
    print("解得： a = ", u[0], " b = ", u[1])  # X = 0.0023 * z + 72.657

    # 求微分方程通解
    t = sp.symbols('t')  # 定义函数和变量
    y = sp.Function('Xt')(t)
    ode = sp.Eq(y.diff(t) + u[0].item() * y, u[1].item())  # 定义微分方程
    solution = sp.dsolve(ode)
    general_solution = solution.rhs
    print("微分方程的通解为：", general_solution)

    # 求微分方程特解 y(0) = 71.1 带入通解方程，可以求得 C
    init_condition = {y.subs(t, 0): x0[1].item()}
    particular_solution = sp.dsolve(ode, ics=init_condition).rhs
    print("微分方程组的解为：", particular_solution)

    Y_list = [particular_solution.subs(t, x_) for x_ in range(n + 1)]
    fit_val = np.diff(Y_list)  # 差分，得到原始序列的拟合值
    print(fit_val)

    # 预测第8年的噪音
    predict = particular_solution.subs(t, 7) - particular_solution.subs(t, 6)
    print(predict)

    # 检验结果
    delta = np.abs(x0[1:] - fit_val.reshape(-1, 1) / x0[1:])
    rho = 1 - (1 - 0.5 * u[0].item()) / (1 + 0.5 * u[0].item()) * lambda_val
    print("delta: ", delta)
    print("rho: ", rho)
