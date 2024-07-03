# coding=utf-8
import pandas as pd
import numpy as np
import random
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

# 读取数据
path = "data/附件4/Ins4_20_40_10.xlsx"

df1 = pd.read_excel(path)
df1.columns = [i for i in range(1, 4)]

df2 = pd.read_excel(path, sheet_name="包装种类及其所需墨盒")
df2.columns = [i for i in range(1, 3)]

df3 = pd.read_excel(path, sheet_name="墨盒切换时间")
df3.columns = [i for i in range(len(df3.columns))]

# 提取数据
num_cartridges = df1[2].count()
num_slots = df1[3].count()
num_pkg_type = df1[1].count()

sw_time = df3.values.tolist()  # 切换时间
for lst in sw_time:
    lst.pop(0)

pkg_type = []
for _, row in df2.iterrows():
    pkg_type.append([int(x) for x in row[2][1:-1].split(", ")])
print(pkg_type)
# ALNS 参数
max_iterations = 1000
initial_temp = 1000
cooling_rate = 0.995
min_temp = 1e-3


def calculate_total_switching_time(sequence, sw_time, pkg_type):
    total_time = 0
    for i in range(len(sequence) - 1):
        for s in range(num_slots):
            if s >= len(pkg_type[sequence[i + 1]]) or s >= len(pkg_type[sequence[i]]):
                break
            curr_cartridge = pkg_type[sequence[i]][s]
            next_cartridge = pkg_type[sequence[i + 1]][s]
            if curr_cartridge != next_cartridge:
                total_time += sw_time[curr_cartridge - 1][next_cartridge - 1]
    return total_time


def alns(sw_time, pkg_type, num_pkg_type, num_slots):
    current_solution = list(range(num_pkg_type))
    random.shuffle(current_solution)
    current_cost = calculate_total_switching_time(current_solution, sw_time, pkg_type)
    best_solution = current_solution[:]
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        for _ in range(max_iterations):
            new_solution = current_solution[:]
            i, j = random.sample(range(num_pkg_type), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_cost = calculate_total_switching_time(new_solution, sw_time, pkg_type)

            if new_cost < current_cost or np.exp((current_cost - new_cost) / temp) > random.random():
                current_solution = new_solution[:]
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution[:]
                    best_cost = new_cost

        temp *= cooling_rate

    return best_solution, best_cost


# 求解工具切换问题
best_sequence, best_cost = alns(sw_time, pkg_type, num_pkg_type, num_slots)

# 输出结果
print("最佳作业序列:", best_sequence)
print("最小总切换时间:", best_cost)

# 显示详细的切换信息
print("详细的墨盒切换详情:")
for i in range(len(best_sequence) - 1):
    print(f"从包装 {best_sequence[i] + 1} 到包装 {best_sequence[i + 1] + 1} 的墨盒切换详情:")
    for s in range(num_slots):
        if s >= len(pkg_type[best_sequence[i]]) or s >= len(pkg_type[best_sequence[i + 1]]):
            break
        curr_cartridge = pkg_type[best_sequence[i]][s]
        next_cartridge = pkg_type[best_sequence[i + 1]][s]
        if curr_cartridge != next_cartridge:
            print(f"插槽 {s + 1}: 从墨盒 {curr_cartridge} 切换到墨盒 {next_cartridge}, "
                  f"切换时间: {sw_time[curr_cartridge - 1][next_cartridge - 1]} 分钟")
