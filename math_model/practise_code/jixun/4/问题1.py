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


df = filter_data(df)

d = {}
stockcode_map = {}

for idx, row in df.iterrows():
    name = row['Description']
    num = row['Quantity']
    stock_code = row['StockCode']
    if name not in d:
        d[name] = num
        stockcode_map[name] = stock_code
    else:
        d[name] += num

sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)

print("\n\n", sorted_d[:5])
print("客户购买最多的前五个商品是: ")

for i in sorted_d[:5]:
    product_name = i[0]
    quantity = i[1]
    stock_code = stockcode_map[product_name]
    print(f"\t商品: {product_name}, StockCode: {stock_code}, 数量: {quantity}")

top_products_list = [item[0] for item in sorted_d[:8]]

filtered_data = df[df['Description'].isin(top_products_list)]
pivot_table = pd.pivot_table(
    filtered_data,
    values='Quantity',
    index='CustomerID',
    columns='Description',
    aggfunc='sum',
    fill_value=0
)

correlation_matrix = pivot_table.corr()

output = pd.DataFrame(correlation_matrix)
output.to_excel("output.xlsx")
