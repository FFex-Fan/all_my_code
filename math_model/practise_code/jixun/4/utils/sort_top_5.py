import pandas as pd

file_path = '../data/online_retail.csv'
data = pd.read_csv(file_path)

top_products = data.groupby('Description').agg({'Quantity': 'sum'}).sort_values(by='Quantity', ascending=False).head(5)

top_products.reset_index(inplace=True)

# 筛选出最受欢迎的前五个商品
top_products_list = top_products['Description'].tolist()

# 仅保留包含前五个商品的交易记录
filtered_data = data[data['Description'].isin(top_products_list)]

# 创建一个透视表，行是客户ID，列是商品描述，值是购买数量
pivot_table = pd.pivot_table(filtered_data,
                             values='Quantity',
                             index='CustomerID',
                             columns='Description',
                             aggfunc='sum',
                             fill_value=0)

# 计算商品之间的相关性矩阵
correlation_matrix = pivot_table.corr()

print(correlation_matrix)
