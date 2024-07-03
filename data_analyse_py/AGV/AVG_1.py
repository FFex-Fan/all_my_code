# coding=utf-8
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_excel("data/AGV_1.xlsx").fillna(-1)  # INF 使用 -1 替换
df.columns = [i for i in range(123)]  # 重命名横坐标
G = nx.Graph()  # 创建无向图
print(df.shape)
# 获取有效数据
for i, row in df.iterrows():
    l_tmp = []
    f = i + 1
    for idx, val in row.items():
        if val != float(-1) and type(val) != type('站点1'):
            if G.get_edge_data(f, idx):  # 去重
                continue
            l_tmp.append((f, idx, val))
            G.add_weighted_edges_from(l_tmp)
    print(l_tmp)

# print(G.get_edge_data(122, 121)) # 获取边权值
pos = nx.spring_layout(G, iterations=500)  # 力导向布局算法，可以有效减少边的交叉, 可适当增加 iterations 的次数

plt.figure(figsize=(5, 5))

nx.draw(G, pos, node_size=20, with_labels=False, font_size=8, node_color='#0081CF')

# 标记指定节点
nx.draw_networkx_nodes(G, pos, {1: "s"}, node_size=30, node_color='red')
nx.draw_networkx_nodes(G, pos, {122: "e"}, node_size=30, node_color='blue')

df = pd.read_excel("data/AGV_1.xlsx", sheet_name="无人车数据")
df.columns = [i for i in range(43)]

start_point = [-1]
for i, row in df.iterrows():
    start_point.append(row[3])
    nx.draw_networkx_nodes(G, pos, [row[3]], node_size=30, node_color='#D65DB1')

plt.savefig("connect.png")

plt.show()
