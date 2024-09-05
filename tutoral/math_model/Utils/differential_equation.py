# coding=utf-8
import sympy as sp

# 定义符号
x = sp.symbols('x')
y = sp.Function('y')(x)

# 定义方程
P = 2
Q = sp.exp(x)
ode = sp.Eq(sp.Derivative(y, x) + P*y, Q)

# 求解通解
general_solution = sp.dsolve(ode)

# 特解需要初始条件，例如 y(0) = 1
initial_condition = {y.subs(x, 0): 1}
particular_solution = sp.dsolve(ode, ics=initial_condition).rhs

Y_list = [particular_solution.subs(x, x_) for x_ in range(10)]

print(general_solution, particular_solution)
print(Y_list)