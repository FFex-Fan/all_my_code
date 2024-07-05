from matplotlib import pyplot as plt
import random
# import matplotlib
from matplotlib import font_manager

# font = {'family': 'Microsoft YaHei',
#         'weight': 'bold'
#         }
# matplotlib.rc("font", **font)

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

y = [random.randint(20, 35) for i in range(120)]
x = [i for i in range(1, 121)]

print(y)

plt.figure(figsize=(20, 10), dpi=80)
plt.plot(x, y)

_x_label = ["你好, {}".format(i) for i in x[::5]]
plt.xticks(x[::5], _x_label, rotation=45, fontproperties=my_font)
plt.yticks(range(min(y), max(y) + 1))

plt.xlabel("时间", fontproperties=my_font)
plt.ylabel("温度", fontproperties=my_font)
plt.title("10点到12点每分钟的气温变化的情况", fontproperties=my_font)

plt.show()
