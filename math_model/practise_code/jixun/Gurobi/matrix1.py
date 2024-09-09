"""
    使用矩阵方式，解决简单的线性约束问题
"""
# using the matrix API:
#  maximize
#        x +   y + 4 z
#  subject to
#        x + 3 y + 3 z <= 4
#      2 x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

try:
    # Create a new model
    m = gp.Model("matrix1")

    # 创建一个决策变量向量，形状为 (3, )，相当于创建了三个二进制决策变量：x[0]、x[1] 和 x[2]，取值只能是 0 或 1。
    x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")

    # 目标函数定义为系数数组 obj（值为 [1.0, 1.0, 2.0]）与变量向量 x 的点积。
    # 目标是最大化函数 1*x[0] + 1*x[1] + 2*x[2]，模型会寻找能最大化该表达式的 x 值。
    obj = np.array([1.0, 1.0, 4.0])
    m.setObjective(obj @ x, GRB.MAXIMIZE)


    val = np.array([1.0, 3.0, 3.0, -2.0, -1.0]) # 提取出系数矩阵，从左到右从上到下将其排成一行
    row = np.array([0, 0, 0, 1, 1]) # 非零元素所在的行，（1.0, 2.0, 3.0）都位于第 0 行，（-1.0, -1.0）位于第 1 行。
    col = np.array([0, 1, 2, 0, 1]) # 非零元素所在的列，（1.0, 2.0, 3.0）分别对应（0， 1， 2）列，（-1.0, -1.0）对应（0， 1）列

    A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

    # rhs 数组表示约束的右边界值，也就是我们在约束条件中设定的常数
    rhs = np.array([4.0, -1.0])

    # 	这里使用矩阵乘法 A @ x 来表示所有的线性约束，将其与 rhs 数组一起设置为不等式约束。
    # 约束条件为：
    # 	1.	  x[0] + 2x[1] + 3x[2] <=  4
    # 	2.	 -x[0] -  x[1]         <= -1 ，这等价于  x[0] + x[1] >= 1 。
    m.addConstr(A @ x <= rhs, name="c")

    # Optimize model
    m.optimize()

    print(x.X)
    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")