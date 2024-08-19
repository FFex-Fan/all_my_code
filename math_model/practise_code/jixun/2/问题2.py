import pandas as pd
import numpy as np

file_name = "data/p1_单柜.xlsx"
single_df = pd.read_excel(file_name)

double_df = pd.read_excel("data/p1_二并柜.xlsx")
triple_df = pd.read_excel("data/p1_三并柜.xlsx")

single_load_time = single_df['Time'].to_numpy()
single_load_data = single_df['Load'].to_numpy()

double_load_time = double_df['Time'].to_numpy()
double_load_data = double_df['Load'].to_numpy()

triple_load_time = triple_df['Time'].to_numpy()
triple_load_data = triple_df['Load'].to_numpy()


def avg_data(data, step):
    lst = []
    group, tmp = 1, 0
    for idx, cur in enumerate(data):
        tmp += cur
        group += 1
        if group % step == 0:
            lst.append(tmp / step)
            tmp = 0
    return lst


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

print(len(mean_label))
print(mean_tm)
print(mean_label)


def optimize_strategy(type_name, data):
    storage_level = 0
    total_cost = 0
    discharge_revenue = 0
    cn, strategy = [], []

    for idx, cur in enumerate(data):
        category = mean_label[idx]
        price = time_to_prices[category]
        load = cur

        if category == 'Peak':
            if storage_level > 0:
                discharge = min(storage_params[type_name]['discharge_power'], storage_level)
                # print("discharge: ", discharge)
                storage_level -= discharge
                discharge_revenue += discharge * price

                # if load < discharge:
                #     load = 0
                #     discharge_revenue += load * price
                #     storage_level -= load
                #     strategy.append(f"Discharge in {idx}")
                # else:
                #     storage_level -= discharge
                #     discharge_revenue += discharge * price
                #     load -= discharge
            # if load > 0:
            # total_cost += load * price

        elif category == 'Valley':
            if storage_level <= storage_params[type_name]['capacity']:
                charge = min(storage_params[type_name]['capacity'] - storage_level,
                             storage_params[type_name]['charge_power'])
                # print("charge: ", charge)
                storage_level += charge
                total_cost += charge * price
                strategy.append(f"Charge in {idx}")
            # total_cost += load * price

        else:
            # total_cost += load * price
            strategy.append(f"None in {idx}")
        # print("cost: ", total_cost)
        # print("revenue: ", discharge_revenue)
        cn.append(discharge_revenue - total_cost)
    return total_cost, strategy, discharge_revenue, cn


single_cost, single_strategy, single_revenue, single_cn = optimize_strategy("single", single_load_data)
double_cost, double_strategy, double_revenue, double_cn = optimize_strategy("double", double_load_data)
triple_cost, triple_strategy, triple_rebenue, triple_cn = optimize_strategy("triple", triple_load_data)

print("单柜优化后总收益为：", round(single_revenue - single_cost, 2))
print("二并柜优化后总收益为：", round(double_revenue - double_cost, 2))
print("三并柜优化后总收益为：", round(triple_rebenue - triple_cost, 2))

single_out = pd.DataFrame(single_cn)
double_out = pd.DataFrame(double_cn)
triple_out = pd.DataFrame(triple_cn)

single_out.columns = ['CN_问题2']
double_out.columns = ['CN_问题2']
triple_out.columns = ['CN_问题2']

single_out.to_excel("data/p2_单柜.xlsx", index=False)
double_out.to_excel("data/p2_二并柜.xlsx", index=False)
triple_out.to_excel("data/p2_三并柜.xlsx", index=False)
