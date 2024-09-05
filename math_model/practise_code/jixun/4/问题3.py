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

# 创建一个数据框，其中每个商品对应的购买者ID集合
item_customer_df = data.groupby('Description')['CustomerID'].unique().reset_index()
print(item_customer_df)

G = nx.Graph()

for _, row in item_customer_df.iterrows():
    customers = row['CustomerID']

    if len(customers) > 1:
        for customer_pair in combinations(customers, 2):
            G.add_edge(customer_pair[0], customer_pair[1])

print(f"图中的节点数: {G.number_of_nodes()}")
print(f"图中的边数: {G.number_of_edges()}")

dc = nx.degree_centrality(G)
bc = nx.betweenness_centrality(G)
cc = nx.closeness_centrality(G)
ec = nx.eigenvector_centrality(G)

centrality_data = {
    'Node': list(dc.keys()),
    'Degree Centrality': list(dc.values()),
    'Betweenness Centrality': list(bc.values()),
    'Closeness Centrality': list(cc.values()),
    'Eigenvector Centrality': list(ec.values())
}

output = pd.DataFrame(centrality_data)

# 将 DataFrame 写入 Excel 文件
output_file = 'centrality_measures.xlsx'
output.to_excel(output_file, index=False)

print(f"网络密度为：{nx.density(G)}")
print(f"网络平均集聚系数:{nx.average_clustering(G)}")
print(f"网络平均聚类系数: {nx.average_clustering(G)}")

# 计算度中心性
# degree_centrality = nx.degree_centrality(G)
# top_5_customers_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# print("度中心性最高的前五个客户:", top_5_customers_degree)
#
# # 计算加权度中心性
# weighted_degree_centrality = nx.degree_centrality(G)
# top_5_customers_weighted = sorted(weighted_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# print("加权度中心性最高的前五个客户:", top_5_customers_weighted)
