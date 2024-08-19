"""
target:
    Maximize  = 1000 * x1 + 2000 * x2 + 3000 * x3

condition:
    x1 + 2 * x2 + 3 * x3 <= 10
    0 * x1 + 1 * x2 + 2 * x3 <= 5
    x1, x2, x3 >= 0
"""
from pulp import *

prob = LpProblem("p1", LpMaximize)

# 定义变量，下限为 0 ， 没有上限
x1 = LpVariable("x1", 0, None, cat=LpContinuous)
x2 = LpVariable("x2", 0, None, cat=LpContinuous)
x3 = LpVariable("x3", 0, None, cat=LpContinuous)

# 添加 max 等式
prob += 1000 * x1 + 2000 * x2 + 3000 * x3

# 添加条件约束
prob += x1 + 2 * x2 + 3 * x3 <= 10
prob += 0 * x1 + x2 + 2 * x3 <= 5

prob.writeLP("p1.lp")
prob.solve()

print("Status:", LpStatus[prob.status]) # 打印是否已经解决（Optimal）

# 打印每个变量求解后的值
for _ in prob.variables():
    print("\t", _.name, " = ", _.varValue)

# 输出最大利润
print("Maximun = ", value(prob.objective))
