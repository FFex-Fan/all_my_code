# coding=utf-8
import matplotlib.pyplot as plt
from matplotlib import font_manager

a = ["战狼2", "速度与激情6", "功夫瑜伽", "西游伏妖篇", "变形金刚5"]
b = [50.01, 26.94, 19.32, 14.24, 12.19]

# 设置图形大小
plt.figure(figsize=(20, 10), dpi=80)

# 字体
myfont = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

# 绘制条形图
# plt.bar(range(len(a)), b, width=0.2, color='orange')
plt.barh(range(len(a)), b, height=0.2)

# 设置字符串到 x 轴
# plt.xticks(range(len(a)), a, rotation=45, fontproperties=myfont)
plt.yticks(range(len(a)), a, fontproperties=myfont)

plt.show()
