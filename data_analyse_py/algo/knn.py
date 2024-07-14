# coding=utf-8
import random
import pandas as pd
import csv

""" 影响因素：
    1. 使用什么方法求距离
    2. k 值的选取 
"""

labels = ("radius", "texture", "perimeter", "area", "smoothness", "compactness", "symmetry", "fractal_dimension")


def get_max_min_val():
    d = {}
    df = pd.read_csv("../data/Prostate_Cancer.csv")

    for key in labels:
        d[key] = (df[key].max(), df[key].min())
    # print(d)
    return d


def distance(d1, d2, max_min_dict):
    res = 0
    for key in labels:  # max - min 归一化
        d1_val, d2_val = float(d1[key]), float(d2[key])
        max_val, min_val = max_min_dict[key][0], max_min_dict[key][1]
        # print("max ----- min", max_val, min_val)
        bottom = max_val - min_val
        d1_true_val = (d1_val - min_val) / bottom
        d2_true_val = (d2_val - min_val) / bottom
        res += (d1_true_val - d2_true_val) ** 2
    return res ** 0.5


def KNN(k, data, train_set, max_min_dic):
    # 1. 计算距离
    res = [
        {"result": train['diagnosis_result'], "distance": distance(data, train, max_min_dic)}
        for train in train_set
    ]

    # 2. 按关键字从小到大排序
    res = sorted(res, key=lambda x: x['distance'])

    # 3. 取前 k 个
    get_k = res[:k]
    # print(get_k)

    # 4. 加权平均
    res_average = {'B': 0, 'M': 0}
    """ plan1 -> (计算加权平均) """
    s = 0
    for item in get_k:
        s += item['distance']
    for item in get_k:
        res_average[item['result']] += 1 - item['distance'] / s

    """
        plan2 -> (直接统计数据)
    """
    # for item in get_k:
    #     res_average[item['result']] += 1

    # print(res_average)
    # print(data['diagnosis_result'])

    if res_average['B'] > res_average['M']:
        return 'B' == data['diagnosis_result']
    return 'M' == data['diagnosis_result']


if __name__ == '__main__':
    max_min_dic = get_max_min_val()
    # 读取数据
    with open('../data/Prostate_Cancer.csv', 'r') as file:
        reader = csv.DictReader(file)
        datas = [i for i in reader]

    # 分组(训练集 + 测试集)
    random.shuffle(datas)
    n = len(datas) // 3

    test_set = datas[:n]
    train_set = datas[n:]

    k = 4 # 设置 k 值

    cnt = 0
    for i in range(len(test_set)):
        cnt += KNN(k, test_set[i], train_set, max_min_dic)

    print("正确次数: ", cnt, "\t 总次数: ", len(test_set))
    print("正确率为: {:.5f}%".format(100 * cnt / len(test_set)))
