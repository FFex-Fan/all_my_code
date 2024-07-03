import pandas as pd
import numpy as np
import random

# 读取数据
path = "data/附件2/Ins1_5_10_3.xlsx"

df1 = pd.read_excel(path)
df1.columns = [i for i in range(1, 4)]

df2 = pd.read_excel(path, sheet_name="包装种类及其所需墨盒")
df2.columns = [i for i in range(1, 3)]

df3 = pd.read_excel(path, sheet_name="墨盒切换时间")
df3.columns = [i for i in range(len(df3.columns))]

# 提取数据
num_cartridges = df1[2].count()
num_slots = df1[3].count()
num_pkg_types = df1[1].count()

switch_time = df3.values.tolist()
for lst in switch_time:
    lst.pop(0)

packaging_requirements = []
for _, row in df2.iterrows():
    packaging_requirements.append([int(x) for x in row[2][1:-1].split(", ")])

print(num_cartridges, num_pkg_types, num_slots)
print(packaging_requirements)
print(switch_time)


# 给定的包装顺序，从1开始
given_order = [1, 2, 3, 4, 5]


def calculate_switch_cost(current, next):
    """计算从当前墨盒到下一个墨盒的切换成本"""
    return switch_time[current][next]


def total_switch_cost(order):
    """计算总的切换成本"""
    total_cost = 0
    for i in range(len(order) - 1):
        current_pack = order[i] - 1
        next_pack = order[i + 1] - 1
        current_inks = packaging_requirements[current_pack]
        next_inks = packaging_requirements[next_pack]

        # 找出需要切换的墨盒
        switches_needed = []
        for next_ink in next_inks:
            if next_ink not in current_inks:
                best_switch_cost = min(
                    calculate_switch_cost(c - 1, next_ink - 1) for c in current_inks if c not in next_inks)
                switches_needed.append(best_switch_cost)

        total_cost += sum(switches_needed)
    return total_cost


def calculate_detailed_switching(order):
    """计算详细的切换过程"""
    switch_details = []
    for i in range(len(order) - 1):
        current_pack = order[i]
        next_pack = order[i + 1]
        current_inks = packaging_requirements[current_pack - 1]
        next_inks = packaging_requirements[next_pack - 1]
        step_details = []

        # 找出需要切换的墨盒
        for next_ink in next_inks:
            if next_ink not in current_inks:
                best_switch = None
                best_switch_cost = float('inf')
                for current_ink in current_inks:
                    if current_ink not in next_inks:
                        cost = calculate_switch_cost(current_ink - 1, next_ink - 1)
                        if cost < best_switch_cost:
                            best_switch_cost = cost
                            best_switch = (current_ink, next_ink)
                step_details.append((best_switch[0], best_switch[1], best_switch_cost))

        switch_details.append((current_pack, next_pack, step_details))
    return switch_details


# 计算总切换成本
total_cost = total_switch_cost(given_order)
print("最小总切换时间:", total_cost)

# 计算详细的切换过程
switch_details = calculate_detailed_switching(given_order)

# 打印详细的包装切换过程
print("包装种类的切换序列及墨盒切换详情：")
for step, (current_pack, next_pack, details) in enumerate(switch_details, 1):
    print(f"步骤 {step}: 从包装种类 {current_pack} 切换到 包装种类 {next_pack}")
    for current_ink, next_ink, time in details:
        print(f"  墨盒 {current_ink} 切换到 墨盒 {next_ink}, 时间: {time} 分钟")