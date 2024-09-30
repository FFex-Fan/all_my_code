# coding=utf-8

"""
target:
    Maximize  = 1000 * x1 + 2000 * x2 + 3000 * x3

condition:
    x1 + 2 * x2 + 3 * x3 <= 10
    0 * x1 + 1 * x2 + 2 * x3 <= 5
    x1, x2, x3 >= 0
"""

from pulp import *

paints = {"type1", "type2", "type3"}
profit = {
    "type1": 1000,
    "type2": 2000,
    "type3": 3000
}

M1 = {
    "type1": 1,
    "type2": 2,
    "type3": 3
}

M2 = {
    "type1": 0,
    "type2": 1,
    "type3": 2
}

# 定义问题名字，求最大值
prob = LpProblem("p2", LpMaximize)

# 定义变量
var = LpVariable.dicts("", paints, 0, None, LpContinuous)

# print(var)
# 添加目标函数
prob += lpSum([profit[i] * var[i] for i in paints])
prob += lpSum([M1[i] * var[i] for i in paints]) <= 10
prob += lpSum([M2[i] * var[i] for i in paints]) <= 5

prob.writeLP("p2.lp")

prob.solve()
print("Status: ", LpStatus[prob.status])

for _ in prob.variables():
    print("\t", _.name, " = ", _.varValue)

print("Maximun = ", value(prob.objective))
