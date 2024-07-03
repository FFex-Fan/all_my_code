# coding=utf-8
from matplotlib import pyplot as plt
from matplotlib import font_manager

a = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
b = [i for i in range(11, 31)]
c = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

plt.figure(figsize=(20, 10), dpi=80)
plt.plot(b, a, label='自己', linestyle='-.')
plt.plot(b, c, label='同桌', linestyle=':')

_x_labels = ['{}岁'.format(i) for i in b]
plt.yticks(range(min(a), max(a) + 1))
plt.xticks(b, _x_labels, rotation=45, fontproperties=my_font)

plt.xlabel("年龄", fontproperties=my_font)
plt.ylabel("交朋友的数量 单位（个）", fontproperties=my_font)
plt.title("年龄与交朋友数", fontproperties=my_font)

# 绘制网格
plt.grid(alpha=0.2)

# 添加图例
plt.legend(prop=my_font, loc='upper left')

plt.show()
