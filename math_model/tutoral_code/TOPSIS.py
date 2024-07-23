import numpy as np


# 区间属性变换
def interval_change(interval, low, up, x):
    a1, a2 = interval
    if a1 <= x <= a2:
        return 1
    elif low <= x <= a1:
        return 1 - (a1 - x) / (a1 - low)
    elif a2 <= x <= up:
        return 1 - (x - a2) / (up - a2)
    else:
        return 0


def normal_vector(a, n, j):
    s = 0
    for i in range(n):
        s += a[i, j] ** 2
    s **= 0.5
    for i in range(n):
        a[i, j] = a[i, j] / s

    # a[:, j] = a[:, j] / np.linalg.norm(a[:, j])


if __name__ == '__main__':
    a = np.array([
        [0.1, 5, 5000, 4.7],
        [0.2, 6, 6000, 5.6],
        [0.4, 7, 7000, 6.7],
        [0.9, 10, 10000, 2.3],
        [1.2, 2, 400, 1.8]
    ])
    print("初始矩阵：\n", a)
    n, m = a.shape

    interval = [5, 6]
    low, up = 2, 12

    # 对师生比进行变换
    for i in range(n):
        a[i, 1] = interval_change(interval, low, up, a[i, 1])

    print("师生比转换后: \n", a)
    # 向量规范化
    for i in range(m):
        normal_vector(a, n, i)

    print("规范化后矩阵为：\n", a)
    b = a * a
    print("按列求和：", b.sum(axis=0))

    w = np.array([0.2, 0.3, 0.4, 0.1])

    # fin_a = np.dot(a, w.T)
    # print(fin_a)

    exp_w = np.tile(w, (5, 1))
    print("拓展权重向量后：\n", exp_w)
    fin_a = a * exp_w
    print("原矩阵加权后：\n", fin_a)

    positive_solution = np.max(fin_a, axis=0)
    print("pre positive ideal solution: ", positive_solution)
    positive_solution[3] = np.min(fin_a[:, 3])
    print("after positive ideal solution: ", positive_solution)

    negative_solution = np.min(fin_a, axis=0)
    negative_solution[3] = np.max(fin_a[:, 3])
    print("negative ideal solution: ", negative_solution)

    pos_dis, neg_dis = [], []
    for i in range(n):
        pos_dis.append(np.linalg.norm(fin_a[i, :] - positive_solution))
        neg_dis.append(np.linalg.norm(fin_a[i, :] - negative_solution))
    pos_dis, neg_dist = np.array(pos_dis), np.array(neg_dis)

    f = neg_dis / (pos_dis + neg_dis)
    print(f)
    res = np.argsort(f)[::-1]
    print("排序结果为：", f[res])
    print("结果的索引为：", res + 1)
