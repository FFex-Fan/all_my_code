# coding=utf-8


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

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# 提取年月信息
data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')

monthly_sales = data.groupby(['YearMonth', 'Description'])['Quantity'].sum().reset_index()

first_five_months = monthly_sales[monthly_sales['YearMonth'] >= '2011-08']

print(first_five_months)

sma_predictions = first_five_months.groupby('Description')['Quantity'].mean().reset_index()
sma_predictions.columns = ['Description', 'Predicted_Quantity']

# 输出预测结果
sma_predictions = sma_predictions.sort_values(by='Predicted_Quantity', ascending=False).head(8)
print("基于简单移动平均的预测：")
print(sma_predictions)

top_products_list = sma_predictions['Description'].tolist()

filtered_data = data[data['Description'].isin(top_products_list)]
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
output.to_excel("output2.xlsx")
