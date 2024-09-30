# Solve the classic diet model, showing how to add constraints
# to an existing model.

import gurobipy as gp
from gurobipy import GRB

# Nutrition guidelines, based on
# USDA Dietary Guidelines for Americans, 2005
# http://www.health.gov/DietaryGuidelines/dga2005/

"""
    (keys, dict1, dict2) = multidict( {
             'key1': [1, 2],
             'key2': [1, 3],
             'key3': [1, 4] 
         } )
返回的结果中：
    keys  值为：['key1', 'key2', 'key3'] # key 的列表

    dict1 值为：{'key1': 1, 'key2': 1, 'key3': 1} # key-val 对应的字典
    dict2 值为：{'key1': 2, 'key2': 3, 'key3': 4}
"""

categories, minNutrition, maxNutrition = gp.multidict(
    {
        "calories": [1800, 2200],
        "protein": [91, GRB.INFINITY],
        "fat": [0, 65],
        "sodium": [0, 1779],
    }
)

# 每种食物对应不同的花费
foods, cost = gp.multidict(
    {
        "hamburger": 2.49,
        "chicken": 2.89,
        "hot dog": 1.50,
        "fries": 1.89,
        "macaroni": 2.09,
        "pizza": 1.99,
        "salad": 2.49,
        "milk": 0.89,
        "ice cream": 1.59,
    }
)

# 各种食物中各种营养成分对应的含量
nutritionValues = {
    ("hamburger", "calories"): 410,
    ("hamburger", "protein"): 24,
    ("hamburger", "fat"): 26,
    ("hamburger", "sodium"): 730,
    ("chicken", "calories"): 420,
    ("chicken", "protein"): 32,
    ("chicken", "fat"): 10,
    ("chicken", "sodium"): 1190,
    ("hot dog", "calories"): 560,
    ("hot dog", "protein"): 20,
    ("hot dog", "fat"): 32,
    ("hot dog", "sodium"): 1800,
    ("fries", "calories"): 380,
    ("fries", "protein"): 4,
    ("fries", "fat"): 19,
    ("fries", "sodium"): 270,
    ("macaroni", "calories"): 320,
    ("macaroni", "protein"): 12,
    ("macaroni", "fat"): 10,
    ("macaroni", "sodium"): 930,
    ("pizza", "calories"): 320,
    ("pizza", "protein"): 15,
    ("pizza", "fat"): 12,
    ("pizza", "sodium"): 820,
    ("salad", "calories"): 320,
    ("salad", "protein"): 31,
    ("salad", "fat"): 12,
    ("salad", "sodium"): 1230,
    ("milk", "calories"): 100,
    ("milk", "protein"): 8,
    ("milk", "fat"): 2.5,
    ("milk", "sodium"): 125,
    ("ice cream", "calories"): 330,
    ("ice cream", "protein"): 8,
    ("ice cream", "fat"): 10,
    ("ice cream", "sodium"): 180,
}

# Model
m = gp.Model("diet")

# 为每种食物创建一个变量，变量 buy[f] 表示购买 f 食物的数量
buy = m.addVars(foods, name="buy")
# You could use Python looping constructs and m.addVar() to create
# these decision variables instead.  The following would be equivalent
#
# buy = {}
# for f in foods:
#   buy[f] = m.addVar(name=f)


# 设置目标函数为：最小化食物总成本
# buy.prod(cost) 相当于每种食物购买数量与其成本的乘积之和
m.setObjective(buy.prod(cost), GRB.MINIMIZE)

# Using looping constructs, the preceding statement would be:
#
# m.setObjective(sum(buy[f]*cost[f] for f in foods), GRB.MINIMIZE)


"""
addConstrs() 是 Gurobi 的一个方法，用于批量添加约束。
它接受一个生成器表达式，生成多个约束，并依次将它们添加到模型 m 中。
第二个参数 "_" 是约束的名称前缀。因为每个约束对应不同的营养类别，它会自动附加编号来区分不同的约束。
"""
m.addConstrs(
    (
        # gp.quicksum(nutritionValues[f, c] * buy[f] for f in foods) 计算所选食物在营养类别 c 上的总含量。
        # gp.quicksum(...) 会对所有食物 f 进行求和，表示在类别 c 上所有食物的总营养量。
        # == [minNutrition[c], maxNutrition[c]] 是约束条件，要求营养类别 c 的总摄入量必须在 minNutrition[c] 和 maxNutrition[c] 之间。
        gp.quicksum(nutritionValues[f, c] * buy[f] for f in foods) == [minNutrition[c], maxNutrition[c]] for c in categories
    ),
    "_",
)


# Using looping constructs, the preceding statement would be:
#
# for c in categories:
#  m.addRange(sum(nutritionValues[f, c] * buy[f] for f in foods),
#             minNutrition[c], maxNutrition[c], c)


def printSolution():
    if m.status == GRB.OPTIMAL:
        print(f"\nCost: {m.ObjVal:g}")
        print("\nBuy:")
        for f in foods:
            if buy[f].X > 0.0001:
                print(f"{f} {buy[f].X:g}")
    else:
        print("No solution")


# Solve
m.optimize()
printSolution()

print("\nAdding constraint: at most 6 servings of dairy")
m.addConstr(buy.sum(["milk", "ice cream"]) <= 6, "limit_dairy")

# Solve
m.optimize()
printSolution()
