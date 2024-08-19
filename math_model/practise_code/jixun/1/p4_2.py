import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

df = pd.read_excel("../data/output.xlsx")
my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

# 常数
C_initial = 0.6  # 初始余氯浓度
C_final = 0.3  # 最终余氯浓度
time_decay = 1.5  # 时间，小时
# 计算比例常数 k
k = np.log(C_initial / C_final) / time_decay
print("k = ", k)
time_step = 0.1  # 数值解的时间步长，单位为小时

# 从加载的数据中提取9月9日的温度数据
temperature_9_9 = df['9_9'].values[9:21]

time_points = np.arange(0, len(temperature_9_9), time_step)


def get_C(t, T_water):
    max_exp_arg = 700  # 经验值，可以根据实际情况调整
    exp_arg = -k * t * (10 ** ((T_water - 25) / 5))
    exp_arg = np.clip(exp_arg, None, max_exp_arg)
    res = C_initial * np.exp(exp_arg)
    print(res)
    return res


def get_T(c, T_water):
    return np.log(1.6667 * c) / (-k * (10 ** ((T_water - 25) / 5)))


leave_chlorines = []
add_times = []

lab = -1
cur_t = 0
total_time = 0
cur_chlorine = C_initial
for i, t in enumerate(time_points):
    cur = i // 12
    T_water = temperature_9_9[cur]

    if lab == -1:
        lab = cur
    if lab != cur:
        cur_t = get_T(cur_chlorine, T_water)
        lab = cur

    cur_chlorine = get_C(cur_t, T_water)
    leave_chlorines.append(cur_chlorine)
    if cur_chlorine <= 0.3:
        add_times.append(9 + t)
        cur_chlorine = C_initial
        cur_t = 0
        total_time += 1
        continue
    cur_t += time_step

print(leave_chlorines)
print("添加氯的时间为:", end=" ")
for i in add_times:
    print(i, end=" ")
print("总添加次数为：", total_time)

# 设置绘图风格
sns.set(style="whitegrid")

# 创建绘图
plt.figure(figsize=(14, 7))
sns.lineplot(x=time_points + 9, y=leave_chlorines, color='#f3b23e', linewidth=2.5)

# 设置标签和标题
plt.xlabel('时间', fontsize=12, labelpad=15, fontproperties=my_font)
plt.ylabel('余氯浓度', fontsize=12, labelpad=15, fontproperties=my_font)
plt.title('泳池中余氯浓度值变化曲线', fontsize=18, fontweight='bold', pad=20, fontproperties=my_font)

# 显示网格
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# 设置坐标轴刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 设置背景颜色
plt.gca().set_facecolor('#f5f5f5')

# 添加图例
# plt.legend(['Data Line'], loc='upper right', fontsize=12)

# 添加边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# 设置边框线宽
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.savefig("../img/p4_2.png")
plt.savefig("../img/p4_2.svg")
# 显示图形
plt.show()
