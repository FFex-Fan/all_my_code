# coding=utf-8

import matplotlib.pyplot as plt

def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    # 获取地图尺寸
    n, m = lines[1].split(" ")
    n, m = int(n), int(m)

    g = []
    for i in range(2, n + 2):
        g.append(lines[i].strip().split())

    num_task = int(lines[n + 3])
    tasks = []
    for i in range(n + 4, n + 4 + num_task):
        sx, sy, ex, ey = map(int, lines[i].split())
        tasks.append((sx, sy, ex, ey))

    return g, tasks, n, m

map_data, tasks, n, m = read_file('../data/附件1/8x8map.txt')


