from matplotlib import pyplot as plt

x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 设置图片
plt.figure(figsize=(15, 10), dpi=80)

# 绘图
plt.plot(x, y)

# 设置 x 轴刻度
# _xtick_size = [float('{:.3f}'.format(i / 2)) for i in range(4, 49)]
_xtick_size = [i / 2 for i in range(4, 49)]
print(_xtick_size)
# plt.xticks(range(2, 25))
plt.xticks(_xtick_size[:: 3])
plt.yticks(range(min(y), max(y) + 1))

# 添加描述信息
# plt.xlabel("")

# 保存
# plt.savefig("./p1.svg")

# 展示
plt.show()
