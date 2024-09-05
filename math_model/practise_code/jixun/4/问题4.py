# coding=utf-8
# coding=utf-8
from itertools import combinations

from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd
import networkx as nx

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

data = data.head(10000)

# 去除空的客户ID
data = data.dropna(subset=['CustomerID'])

# 确保客户ID为整数类型
data['CustomerID'] = data['CustomerID'].astype(int)

item_customer_df = data.groupby('Description')['CustomerID'].apply(list).reset_index()
print(item_customer_df)

item_customer_df.to_excel("customer.xlsx", index=False)

G = nx.Graph()

for _, row in item_customer_df.iterrows():
    customers = row['CustomerID']

    if len(customers) > 1:
        for customer_pair in combinations(customers, 2):
            G.add_edge(customer_pair[0], customer_pair[1])

print(f"图中的节点数: {G.number_of_nodes()}")
print(f"图中的边数: {G.number_of_edges()}")

dc = nx.degree_centrality(G)
top_5_customers = sorted(dc.items(), key=lambda x: x[1], reverse=True)
top_customers = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:5]
print("最重要的前五个客户:", top_customers)

weight = {}
for item in top_5_customers:
    weight[item[0]] = item[1]


important = {}
for _, row in item_customer_df.iterrows():
    description = row['Description']
    customers = row['CustomerID']

    important[description] = 0
    for i in customers:
        if i in weight.keys():
            important[description] += weight[i]


top_important = sorted(important.items(), key=lambda x: x[1], reverse=True)[:5]

print('最重要的五个商品：')
for i in top_important:
    print(i)



