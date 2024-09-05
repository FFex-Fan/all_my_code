import matplotlib
import matplotlib.pyplot as plt

# 地图数据，包含障碍物
map_data = [
    ". . . . . . @ @ @ @ @ @ @ @ @ @".split(),
    ". . @ @ . . . . . . . @ . . @ .".split(),
    ". . . . @ @ . . @ @ . . . @ @ @".split(),
    ". @ . . @ . @ . . . . . . @ . .".split(),
    ". . @ @ . . . . . @ @ @ . . . .".split(),
    ". . . . . @ @ @ . . . . . @ @ @".split(),
    "@ @ @ @ . . . @ . @ @ . @ @ @ @".split(),
    ". . . . . . . @ . . . . . @ @ @".split(),
    ". . @ . . @ @ @ @ @ . . @ . . .".split(),
    ". . @ . . @ . . . @ . . . . @ @".split(),
    ". . . . @ . . . @ . . @ . . @ .".split(),
    ". @ . . @ . @ . @ . @ @ . . . .".split(),
    ". @ . @ . . @ . . . . . . @ @ @".split(),
    ". . . . . . @ . . @ . @ . . . .".split(),
    "@ @ @ @ . @ . . . @ @ @ @ . . .".split(),
    ". . . . . . . . @ . . . . @ @ @".split()
]

# 定义机器人的路径，每个路径是一个列表
robot_paths = [
    [(2, 7), (1, 7), (1, 6)],
    [(10, 1), (10, 2), (10, 3), (9, 3), (8, 3), (7, 3), (7, 4), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (4, 7), (
        4, 8), (5, 8), (6, 8), (7, 8), (7, 9)],
    [(1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1)],
    [(7, 6), (6, 6), (6, 5), (6, 4), (5, 4)],
    [(7, 2), (7, 3), (7, 4), (7, 5), (6, 5)],
    [(6, 6), (6, 5), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (4, 7), (3, 7), (3, 8), (3, 9), (3, 10), (2, 10), (
        2, 11), (2, 12), (1, 12)],
    [(12, 11), (12, 10), (12, 9), (11, 9), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (7, 9), (7, 8), (6, 8), (
        5, 8), (5, 9), (5, 8), (4, 8), (4, 7), (4, 6)],
    [(3, 11), (3, 10), (3, 9), (3, 8), (3, 7), (4, 7), (3, 7), (4, 7), (4, 6), (4, 5)],
    [(14, 6), (15, 6), (15, 5), (15, 4), (14, 4), (13, 4), (13, 3), (13, 2), (12, 2), (11, 2), (10, 2), (10, 3), (
        9, 3), (8, 3), (8, 4), (7, 4), (7, 5), (7, 6)],
    [(4, 5), (4, 4), (5, 4), (5, 3), (5, 2), (5, 1), (4, 1), (4, 0), (3, 0), (2, 0), (2, 1), (2, 2), (3, 2)],
    [(3, 15), (4, 15), (4, 14), (4, 13), (4, 12), (5, 12), (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (9, 10), (
        10, 10), (10, 9), (11, 9), (12, 9), (12, 8), (12, 7), (11, 7), (10, 7), (9, 7), (9, 8)],
    [(14, 14), (14, 13), (13, 13), (13, 12), (12, 12), (11, 12), (10, 12), (9, 12), (9, 11), (9, 10), (8, 10), (
        7, 10), (7, 11), (6, 11)]
]
print(map_data)

# 创建地图并绘制障碍物和路径
fig, ax = plt.subplots(figsize=(12, 12))

# 生成12种配色
colors = plt.cm.get_cmap('tab20', 12).colors
# colors = matplotlib.colormaps.get_cmap()

# 绘制障碍物
for y, row in enumerate(map_data):
    for x, cell in enumerate(row):
        if cell == '@':
            ax.plot(x, len(map_data) - y - 1, 's', color='black', markersize=10)

# 绘制每个机器人的路径
for i, path in enumerate(robot_paths):
    y, x = zip(*path)  # 正确使用y为行，x为列
    y = [len(map_data) - 1 - j for j in y]  # 反转y轴以匹配绘图
    ax.plot(x, y, marker='o', linestyle='-', markersize=5, color=colors[i % len(colors)], label=f'Robot {i + 1}')

    # 添加方向箭头
    for j in range(len(x) - 1):
        ax.arrow(x[j], y[j], x[j + 1] - x[j], y[j + 1] - y[j], head_width=0.15, head_length=0.15,
                 fc=colors[i % len(colors)], ec=colors[i % len(colors)])

# 设置轴的范围、标签和网格
ax.set_xlim(-0.5, len(map_data[0]) - 0.5)
ax.set_ylim(-0.5, len(map_data) - 0.5)
ax.set_xticks(range(len(map_data[0])))
ax.set_yticks(range(len(map_data)))
ax.set_xticklabels(range(len(map_data[0])))  # 横坐标从0开始
ax.set_yticklabels(range(len(map_data) - 1, -1, -1))  # 纵坐标从15到0，从上到下
ax.xaxis.tick_top()  # 将横坐标移到顶部
ax.grid(True)

# 添加图例
ax.legend(loc='upper right', fontsize='small', ncol=2)

plt.savefig("16x16.png")
# 显示带有路径和障碍物的地图
plt.show()