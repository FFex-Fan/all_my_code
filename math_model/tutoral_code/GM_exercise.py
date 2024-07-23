# coding=utf-8

import numpy as np
import sympy as sp

if __name__ == '__main__':
    x0 = np.array([100, 120, 135, 150, 165])
    n = len(x0)
    x1 = np.cumsum(x0)
    print(x1)
    # 构造矩阵 B 和向量 Y
    B = np.hstack((-0.5 * (x1[:n - 1] + x1[1:n]).reshape(-1, 1), np.ones((4, 1))))
    Y = np.array([120, 135, 150, 165]).reshape(-1, 1)
    print(B)
    print(Y)
    u, rediuals, rank, s = np.linalg.lstsq(B, Y, rcond=None)
    u = np.round(u, 5)
    a, b = u[0].item(), u[1].item()
    print("a = ", a, " b = ", b)

    x = sp.symbols('x')
    y = sp.Function('y')(x)

    ode = sp.Eq(sp.Derivative(y, x) + a * x, b)
    general_solution = sp.dsolve(ode).rhs

    print(general_solution)

