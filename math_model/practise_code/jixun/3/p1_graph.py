import matplotlib.pyplot as plt


map_data = [
'. . . . . . . .' .split(),
'. @ . . @ . @ .'.split(),
'. . . . . . . .'.split(),
'. . . . . . . .'.split(),
'@ . . . . . . .'.split(),
'. . . . . . . .'.split(),
'. . . . @ . . .'.split(),
'. @ . . . . . .'.split()
]

# Define the paths for each robot
robot_paths = [
    [(1, 7), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3)],
    [(5, 0), (5, 1), (6, 1), (6, 2), (6, 3), (7, 3), (7, 4), (7, 5)],
    [(7, 4), (7, 3), (6, 3), (5, 3), (5, 2)],
    [(4, 5), (5, 5), (6, 5), (6, 6), (6, 7)],
    [(6, 5), (6, 6), (5, 6), (4, 6), (3, 6), (2, 6), (2, 7), (1, 7), (0, 7), (0, 6)],
    [(5, 2), (5, 3), (4, 3), (3, 3), (3, 4), (2, 4)],
    [(0, 4), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)],
    [(4, 1), (3, 1), (2, 1), (2, 2), (1, 2)]
]




# 创建地图并绘制障碍物和路径
fig, ax = plt.subplots(figsize=(10, 10))

# 生成12种配色
colors = plt.cm.get_cmap('tab20', 8).colors

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

plt.savefig("8x8.png")
# 显示带有路径和障碍物的地图
plt.show()