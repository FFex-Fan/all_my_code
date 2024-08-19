# coding=utf-8
import pandas as pd
import numpy as np

# 初始参数
basic_demand_cost = 48  # 基础需量电价（元/kW）
extra_demand_cost = basic_demand_cost * 2  # 超过105%需量的额外电价（元/kW)
timedelta = 15

# 读取数据
single_df = pd.read_excel("data/p1_单柜.xlsx")
double_df = pd.read_excel("data/p1_二并柜.xlsx")
triple_df = pd.read_excel("data/p1_三并柜.xlsx")

single_load_time = single_df['Time'].to_numpy()
single_load_data = single_df['Load'].to_numpy()

double_load_time = double_df['Time'].to_numpy()
double_load_data = double_df['Load'].to_numpy()

triple_load_time = triple_df['Time'].to_numpy()
triple_load_data = triple_df['Load'].to_numpy()

time_to_prices = {
    'Peak': 1.1353,  # 峰
    'Flat': 0.688,  # 平
    'Valley': 0.3096  # 谷
}
time_to_type = {
    'Peak': [8, 9, 10, 13, 14, 15, 16],  # 峰
    'Flat': [i for i in range(17, 24)],  # 平
    'Valley': [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]  # 谷
}

# 定义储能系统的参数
storage_params = {
    'single': {'capacity': 215, 'charge_power': 101.4, 'discharge_power': 101.4},
    'double': {'capacity': 215 * 2, 'charge_power': 201.5, 'discharge_power': 201.5},
    'triple': {'capacity': 215 * 3, 'charge_power': 308.1, 'discharge_power': 308.1}
}


def assign_price_category(tm):
    if tm in time_to_type['Peak']:
        return 'Peak'
    elif tm in time_to_type['Flat']:
        return 'Flat'
    else:
        return 'Valley'


mean_tm = [int(_[:2]) for _ in single_load_time]
mean_label = [assign_price_category(i) for i in mean_tm]


def calculate_sliding_demand(param, timedelta):
    return param.rolling(window=timedelta, min_periods=1).mean()


def opt(type_name, data, contract_demand):
    storage_level = 0  # 初始化储能系统的电量
    total_charge_cost = 0  # 初始化总充电成本
    total_discharge_revenue = 0  # 初始化总放电收益
    cn, demand = [], []  # 用于记录每个时刻的实际需量（电力需求）

    for idx, cur in enumerate(data):
        category = mean_label[idx]
        price = time_to_prices[category]
        load = cur

        discharge = 0
        charge = 0

        if category == 'Valley':
            if storage_level < storage_params[type_name]['capacity']:
                charge = min(storage_params[type_name]['charge_power'],
                             storage_params[type_name]['capacity'] - storage_level)
                storage_level += charge
                total_charge_cost += charge * price
        elif category == 'Peak':
            if storage_level > 0:
                discharge = min(storage_params[type_name]['discharge_power'], storage_level)
                storage_level -= discharge
                total_discharge_revenue += discharge * price

        current_demand = load + charge - discharge
        demand.append(current_demand)
        cn.append(current_demand)

    # 计算滑动窗口内的需量
    t_demand = calculate_sliding_demand(pd.Series(demand), timedelta)
    max_demand = t_demand.max()

    # 计算需量电费
    if max_demand > 1.05 * contract_demand:
        # 如果最大需量超过合同需量的105%，需要支付额外电费
        demand_cost = basic_demand_cost * contract_demand + extra_demand_cost * (max_demand - 1.05 * contract_demand)
    else:
        # 如果最大需量未超过合同需量的105%，仅支付基础电费
        demand_cost = basic_demand_cost * contract_demand

    # 计算峰谷套利收益
    arbitrage_profit = total_discharge_revenue - total_charge_cost

    total_profit = arbitrage_profit - 0.003 * demand_cost

    return total_profit, arbitrage_profit, demand_cost, cn


def find_optimal_contract_demand(load_data, type_name, demand_range):
    best_profit = -np.inf
    best_contract_demand = 0
    best_arbitrage_profit = 0
    best_demand_cost = 0
    best_cn = 0

    for contract_demand in range(demand_range[0], demand_range[1], demand_range[2]):
        profit, arbitrage_profit, demand_cost, cn = opt(type_name, load_data, contract_demand)
        if profit > best_profit:
            best_profit = profit
            best_contract_demand = contract_demand
            best_arbitrage_profit = arbitrage_profit
            best_demand_cost = demand_cost
            best_cn = cn

    return best_contract_demand, best_profit, best_arbitrage_profit, best_demand_cost, best_cn


demand_range = (300, 2000, 1)  # 合同最大需量的范围

# 优化每个储能系统的合同最大需量
single_best_demand, single_best_profit, single_best_arbitrage, single_best_cost, single_cn = find_optimal_contract_demand(
    single_load_data, "single", demand_range)
double_best_demand, double_best_profit, double_best_arbitrage, double_best_cost, double_cn = find_optimal_contract_demand(
    double_load_data, "double", demand_range)
triple_best_demand, triple_best_profit, triple_best_arbitrage, triple_best_cost, triple_cn = find_optimal_contract_demand(
    triple_load_data, "triple", demand_range)

print(
    f"单柜最优合同最大需量: {single_best_demand} kW, 最优收益: {single_best_profit:.2f}, 峰谷套利收益: {single_best_arbitrage:.2f}, 需量电费: {single_best_cost:.2f}")
print(
    f"二并柜最优合同最大需量: {double_best_demand} kW, 最优收益: {double_best_profit:.2f}, 峰谷套利收益: {double_best_arbitrage:.2f}, 需量电费: {double_best_cost:.2f}")
print(
    f"三并柜最优合同最大需量: {triple_best_demand} kW, 最优收益: {triple_best_profit:.2f}, 峰谷套利收益: {triple_best_arbitrage:.2f}, 需量电费: {triple_best_cost:.2f}")

# 将结果保存到文件中
single_out = pd.DataFrame({"Time": single_load_time, "CN": single_cn})
double_out = pd.DataFrame({"Time": double_load_time, "CN": double_cn})
triple_out = pd.DataFrame({"Time": triple_load_time, "CN": triple_cn})

single_out.to_excel("data/p3_单柜_优化结果.xlsx", index=False)
double_out.to_excel("data/p3_二并柜_优化结果.xlsx", index=False)
triple_out.to_excel("data/p3_三并柜_优化结果.xlsx", index=False)
