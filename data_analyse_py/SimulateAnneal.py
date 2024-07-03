# coding=utf-8
import math
import random
import pandas as pd
from geopy.distance import geodesic

""" 为了方便计算做如下规定：
    0: 出发点（北京）
    1 ~ 33: 需要经过的城市
    34: 终点（北京）
"""
xlsx_file = pd.ExcelFile('data/citys.xlsx')  # 读取xlsx文件中的数据
sheet_names = xlsx_file.sheet_names

data = []
for sheet_name in sheet_names:
    df = pd.read_excel(xlsx_file, sheet_name)
    data.append(df.values)
data = data[0]
print(type(data))  # <class 'numpy.ndarray'>

longitude = []  # 精度
latitude = []  # 纬度
n = len(data)  # 城市的个数
d = [[0 for j in range(n + 1)] for i in range(n)]  # 邻接矩阵用于记录不同点之间的距离

for i in range(n):  # 记录下每个城市的经纬度
    longitude.append(data[i][1])
    latitude.append(data[i][2])

for i in range(n):
    for j in range(n):
        # 根据两点间的经纬度，求出两点之间的距离
        d[i][j] = geodesic((latitude[i], longitude[i]), (latitude[j], longitude[j])).km

for i in range(n):  # 计算各城市最后到达(返回)起点的距离
    d[i][n] = geodesic((latitude[i], longitude[i]), (latitude[0], longitude[0])).km

# for i in range(len(d)):
#     for j in range(len(d[0])):
#         print(d[i][j], end=' ')
#     print()


""" 蒙特卡洛法求初始解 """
path = []
distance = math.inf  # 初始化为正无穷
T = 1
eps = 1e-30
alpha = 0.999

for i in range(1000):  # 随机求 1000 种方案，最好的结果作为初始解
    tmp_path = [_ for _ in range(1, n)]
    random.shuffle(tmp_path)
    tmp_path.insert(0, 0)  # 开头添加 0（表示从起始点开始）
    tmp_path += [n]  # 末尾拼接 34（表示最终需要返回起点）
    tmp_dis = 0  # 记录本条路径需要的距离
    for j in range(len(tmp_path) - 1):
        tmp_dis += d[tmp_path[j]][tmp_path[j + 1]]
    if tmp_dis < distance:  # 若当前结果比记录的结果更优，则更新当前结果，
        distance = tmp_dis
        path = tmp_path
print(path)
print(distance)

""" 模拟退火 """
while T > eps:  # 当 T 趋于 0 时退出循环
    for i in range(1000):  # 马尔科夫过程（在符合均匀分布的前提下，当前解只和上一次的解相关）
        c1 = math.floor(random.random() * 33) + 1  # 取得一点 c1
        c2 = math.floor(random.random() * 33) + 1  # 取得一点 c2
        if c1 > c2:
            c1, c2 = c2, c1  # 保证 c1 <= c2
        tmp = path[c2: c1 - 1: -1]  # 将列表中 c1 ~ c2 的序列进行翻转

        theta = (d[path[c1 - 1]][path[c2]] + d[path[c1]][path[c2 + 1]]  # 计算处 theta 的值
                 - d[path[c1 - 1]][path[c1]] - d[path[c2]][path[c2 + 1]])

        """ 当出现如下两种情况之一时：
                1. theta < 0, 根据 theta 的计算式，说明距离缩短（当前还未达到局部最优，需要接受更优解）
                2. exp(-theta / T) >= 0 ~ 1 的一个随机数（保证不会陷入局部最优）
            则接收新解
            
            解释说明：
                当 T 趋于 0 时（温度极低），由于 theta 不为 0，则 exp(-theta / T) 的值趋向于 0，说明会有更大概率留在当前解
        """
        if theta < 0 or math.exp(-theta / T) >= random.random():
            path = path[:c1] + tmp + path[c2 + 1:]  # 更新成新的路径
            distance += theta
        # break
    T *= alpha
    # break

print(path)
print(distance)
