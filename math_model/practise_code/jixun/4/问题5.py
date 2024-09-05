# coding=utf-8
# coding=utf-8
from itertools import combinations

import networkx as nx
from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd

online_retail = fetch_ucirepo(id=352)

print(type(online_retail.data))

"""
    online_retail.data.ids          # id 数据
    online_retail.data.features     # 特征数据
    online_retail.data.targets      # 目标数据
    online_retail.data.original     # 完整数据
    online_retail.data.headers      # 属性名称
"""

df = pd.DataFrame(online_retail.data.original)


def filter_data(data):
    print("The obs of input data is: ", len(data))
    newdata = data[data['Quantity'] > 0].reset_index(drop=True)
    print("The obs after keeping Quantity > 0 are:", len(newdata))
    newdata = newdata[newdata['CustomerID'].notnull()].reset_index(drop=True)
    print("The obs after keeping non-null CustomerID are:", len(newdata))
    newdata = newdata[newdata['Country'] == 'United Kingdom'].reset_index(drop=True)
    print("The obs after keeping UK records are:", len(newdata))
    newdata = newdata[newdata['UnitPrice'] > 0].reset_index(drop=True)
    print("The obs after keeping non-negative UnitPrice are:", len(newdata))
    newdata = newdata[newdata['StockCode'].map(lambda x: len(str(x))) == 5].reset_index(drop=True)
    print("The obs after keeping 5-digit StockCode are:", len(newdata))
    return newdata


data = filter_data(df)

# 去除空的客户ID
data = data.dropna(subset=['CustomerID'])

# 转换日期格式
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# 提取年、月、日信息
data['Day'] = data['InvoiceDate'].dt.date

# 选择特定时间范围内的数据
start_date = pd.to_datetime('2011-11-09').date()
end_date = pd.to_datetime('2011-12-09').date()
specific_range_data = data[(data['Day'] >= start_date) & (data['Day'] <= end_date)]


# 初始化一个字典来存储每天的网络
daily_networks = {}

# 构建每日的客户关系网络
for day, group in specific_range_data.groupby('Day'):
    G = nx.Graph()
    # 提取每个商品的客户列表
    item_customer_df = group.groupby('Description')['CustomerID'].unique().reset_index()
    for _, row in item_customer_df.iterrows():
        customers = row['CustomerID']
        if len(customers) > 1:  # 至少两个客户购买了该商品
            for customer_pair in combinations(customers, 2):
                if G.has_edge(*customer_pair):
                    G[customer_pair[0]][customer_pair[1]]['weight'] += 1
                else:
                    G.add_edge(customer_pair[0], customer_pair[1], weight=1)

    # 将网络添加到字典中
    daily_networks[day] = G

# 打印每天的网络信息
for day, G in daily_networks.items():
    print(f"\n日期: {day}")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")

network_metrics = []

for day, G in daily_networks.items():
    metrics = {
        '时间': day,
        '节点数': G.number_of_nodes(),
        '边数': G.number_of_edges(),
        '网络平均集聚系数': nx.average_clustering(G),
        '网络密度为': nx.density(G),
        '网络平均聚类系数': nx.average_clustering(G)
    }
    network_metrics.append(metrics)
metrics_df = pd.DataFrame(network_metrics)


metrics_df.to_excel("output5.xlsx", index=False)
# 打印分析结果
print("\n从 2011-11-09 到 2011-12-09 每日网络参数分析：")
print(metrics_df)


