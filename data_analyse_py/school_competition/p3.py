import pandas as pd
import numpy as np
import random

# Ins2_10_30_10 (282, 282, 282, 282)
# Ins3_20_50_10 (457, 479, 478, 476, 473, 473, 459, 460)
# Ins4_30_60_10 (604, 600, 619, 600, 609, 610, 604, 603)


path = ("data/附件3/Ins4_30_60_10.xlsx")

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

sw_time = df3.values.tolist()
for lst in sw_time:
    lst.pop(0)

pkg_type = []
for _, row in df2.iterrows():
    pkg_type.append([int(x) for x in row[2][1:-1].split(", ")])

# 模拟退火参数
initial_temp = 1000
cooling_rate = 0.998
min_temp = 1e-3


def calculate_total_switching_time(sequence, switching_time, packages):
    total_time = 0
    for i in range(len(sequence) - 1):
        for s in range(num_slots):
            curr_cartridge = packages[sequence[i]][s] if s < len(packages[sequence[i]]) else -1
            next_cartridge = packages[sequence[i + 1]][s] if s < len(packages[sequence[i + 1]]) else -1
            if curr_cartridge != -1 and next_cartridge != -1 and curr_cartridge != next_cartridge:
                total_time += switching_time[curr_cartridge - 1][next_cartridge - 1]
    return total_time


def initialize_solution(packages, num_packages, num_slots):
    solution = []
    for pkg in packages:
        solution.append(pkg + [0] * (num_slots - len(pkg)))
    return solution


def get_neighbor(solution, num_packages, num_slots):
    new_solution = [row[:] for row in solution]
    for p in range(num_packages):
        s1, s2 = random.sample(range(num_slots), 2)
        new_solution[p][s1], new_solution[p][s2] = new_solution[p][s2], new_solution[p][s1]
    return new_solution


def simulated_annealing(switching_time, packages, num_packages, num_slots):
    current_solution = initialize_solution(packages, num_packages, num_slots)
    current_cost = calculate_total_switching_time(range(num_packages), switching_time, current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost
    temp = initial_temp

    while temp > min_temp:
        for _ in range(100):
            new_solution = get_neighbor(current_solution, num_packages, num_slots)
            new_cost = calculate_total_switching_time(range(num_packages), switching_time, new_solution)

            if new_cost < current_cost or np.exp((current_cost - new_cost) / temp) > random.random():
                current_solution = new_solution[:]
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution[:]
                    best_cost = new_cost

        temp *= cooling_rate

    return best_solution, best_cost


# 求解工具切换问题
fixed_sequence = list(range(num_pkg_type))
best_solution, best_cost = simulated_annealing(sw_time, pkg_type, num_pkg_type, num_slots)

# 输出结果
print("最优插槽排列:")
for i, pkg in enumerate(best_solution):
    print(f"包装 {i + 1}: 插槽排列 {pkg}")

print("最小总切换时间:", best_cost)

# 显示详细的切换信息
print("详细的墨盒切换详情:")
for i in range(num_pkg_type - 1):
    print(f"从包装 {i + 1} 到包装 {i + 2} 的墨盒切换详情:")
    for s in range(num_slots):
        curr_cartridge = best_solution[i][s]
        next_cartridge = best_solution[i + 1][s]
        if curr_cartridge != next_cartridge and curr_cartridge != 0 and next_cartridge != 0:
            print(
                f"插槽 {s + 1}: 从墨盒 {curr_cartridge} 切换到墨盒 {next_cartridge}, 切换时间: {sw_time[curr_cartridge - 1][next_cartridge - 1]} 分钟")
print("最小总切换时间:", best_cost)