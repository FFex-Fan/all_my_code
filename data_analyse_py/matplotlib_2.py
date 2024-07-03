# coding=utf-8

from matplotlib import pyplot as plt
from matplotlib import font_manager

y3 = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14,
      15, 15, 15, 19, 21, 22, 22, 22, 23]
y10 = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23, 17, 20, 21, 20, 22, 15, 11,
       15, 5, 13, 17, 10, 11, 13, 12, 13, 6]
x3 = [i for i in range(1, 32)]
x10 = [i for i in range(51, 82)]

# 设置图形大小
plt.figure(figsize=(20, 10), dpi=80)

# 绘制散点图
plt.scatter(x3, y3, label='三月')
plt.scatter(x10, y10, label='十月')

# 创建中文对象
my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

# 调整 x 轴刻度
_x = x3 + x10
_xtick = ["3月{}日".format(i) for i in x3]
_xtick += ["10月{}日".format(i - 50) for i in x10]
plt.xticks(_x[::3], _xtick[::3], rotation=45, fontproperties=my_font)

# 添加描述信息
plt.xlabel("时间", fontproperties=my_font)
plt.ylabel("温度", fontproperties=my_font)
plt.title("标题", fontproperties=my_font)

# 添加图例
plt.legend(prop=my_font, loc='upper right')

# 显示
plt.show()
