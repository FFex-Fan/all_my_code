import pandas as pd
import numpy as np

df = pd.read_excel('data/data3-3.xlsx', sheet_name='Sheet1').drop(columns=['学生编号'])
n, m = df.shape
data = np.zeros((n, m))
for j in range(m):
    for i in range(n):
        data[i, j] = (df.iloc[i, j] - np.min(df.iloc[:, j])) / (np.max(df.iloc[:, j]) - np.min(df.iloc[:, j]))
        if data[i, j] == 0:
            data[i, j] = 0.0001

p = data / np.sum(data, axis=0)
e = -np.sum(p * np.log(p), axis=0) / np.log(n)

# e = np.round(e, 5) # 设置保留 5 位小数
# dd = pd.DataFrame(e.reshape(1, -1))
# dd.to_excel('output.xlsx')

d = 1 - e
w = d / np.sum(d)  # 计算每个属性的权重
s = np.dot(p, w.T)

idx = np.argsort(s)[::-1]
print(idx + 1)
print(s[idx])
