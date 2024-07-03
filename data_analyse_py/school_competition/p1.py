import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

path = "data/1/Ins5_30_60_10.xlsx"

df1 = pd.read_excel(path).fillna("none")
df1.columns = [i for i in range(1, 4)]

df2 = pd.read_excel(path, sheet_name="包装种类及其所需墨盒")
df2.columns = [i for i in range(1, 3)]

num_cartridges = 0  # 墨盒数
num_slots = 0  # 插槽数
num_packages = 0  # 包装数

for i, row in df1.iterrows():
    if row[1] != "none":
        num_packages += 1
    if row[2] != "none":
        num_cartridges += 1
    if row[3] != "none":
        num_slots += 1

packages = {}  # 包装种类和对应墨盒需求
for i, row in df2.iterrows():
    lst = row[2][1:-1].split(", ")
    tmp = [int(i) for i in lst]
    packages[row[1]] = tmp

# 定义一个最小化问题，目标：减少墨盒的总切换次数
model = LpProblem("Minimize", LpMinimize)

# 定义变量 x 和 y（该变量取值为 0 或 1），分别表示包装在插槽的墨盒使用情况和切换情况。
# x[p][s][c] 表示包装 p 在插槽 s 使用墨盒 c，如果使用则为1，否则为0。
# y[p][s][c] 表示从包装 p 切换到包装 p+1 时插槽 s 是否由墨盒 c 切换为其他墨盒，如果切换则为1，否则为0。
x = LpVariable.dicts("used", (range(1, num_packages + 1), range(1, num_slots + 1), range(1, num_cartridges + 1)),
                     cat="Binary")
y = LpVariable.dicts("switch", (range(1, num_packages), range(1, num_slots + 1), range(1, num_cartridges + 1)),
                     cat="Binary")

# lpSum 用于计算目标函数，即所有 y[p][s][c] 的和，表示总切换次数。
model += lpSum(
    y[p][s][c] for p in range(1, num_packages) for s in range(1, num_slots + 1) for c in range(1, num_cartridges + 1))

# 约束条件1：每个时刻插槽只能放一个墨盒
# 对于每个包装 p 和每个插槽 s，所有墨盒 c 的变量 x[p][s][c] 的和必须等于 1，即：确保每个插槽在每次印刷时只能放置一个墨盒。
for p in range(1, num_packages + 1):
    for s in range(1, num_slots + 1):
        model += lpSum(x[p][s][c] for c in range(1, num_cartridges + 1)) == 1

# 约束条件2：每个包装所需的所有墨盒必须放置在插槽中
# 对于每个包装 p，确保所需的每个墨盒 c 在某个插槽 s 中出现，即：确保每个包装所需的所有墨盒都能被安排。
for p in range(1, num_packages + 1):
    for c in packages[p]:
        model += lpSum(x[p][s][c] for s in range(1, num_slots + 1)) >= 1

# 约束条件3：切换约束
# 如果在插槽 s 中墨盒从时刻 p 到 p+1 发生更换，即 x[p][s][c] 与 x[p+1][s][c] 不同，y[p][s][c] 将被迫为1，表示切换发生。
for p in range(1, num_packages):
    for s in range(1, num_slots + 1):
        for c in range(1, num_cartridges + 1):
            model += y[p][s][c] >= x[p][s][c] - x[p + 1][s][c]

# 使用PuLP的默认求解器CBC求解问题
model.solve(PULP_CBC_CMD(msg=False))

# 输出结果
total_switches = lpSum(y[p][s][c].varValue for p in range(1, num_packages) for s in range(1, num_slots + 1) for c in
                       range(1, num_cartridges + 1))
print("切换次数:", total_switches)

# 打印详细的切换信息
for p in range(1, num_packages):
    for s in range(1, num_slots + 1):
        for c in range(1, num_cartridges + 1):
            if y[p][s][c].varValue > 0:
                print(f"在 {p} 的插槽 {s} 中从墨盒 {c} 切换")

# 打印墨盒的使用情况
for p in range(1, num_packages+1):
    for s in range(1, num_slots+1):
        for c in range(1, num_cartridges+1):
            if x[p][s][c].varValue > 0:
                print(f"包装 {p} 在插槽 {s} 中使用墨盒 {c}")
