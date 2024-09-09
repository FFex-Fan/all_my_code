"""
    求解双线性优化问题，涉及到非线形约束条件
"""

import gurobipy as gp
from gurobipy import GRB # 常量库

m = gp.Model("bilinear")

x = m.addVar(name='x')
y = m.addVar(name='y')
z = m.addVar(name='z')

# 设置目标函数（最大化 x）即，找到 x 值对大的解
m.setObjective(1.0 * x, GRB.MAXIMIZE)

# 线性约束
m.addConstr(x + y + z <= 10, "c0")

# 非线形约束1
m.addConstr(x * y <= 2, "bilinear0")

# 非线性约束2
m.addConstr(x * z + y * z == 1, "bilinear1")


m.optimize()

m.printAttr("x")