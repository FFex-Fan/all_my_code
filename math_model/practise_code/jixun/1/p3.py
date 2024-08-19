import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

data_1 = pd.read_excel("../data/data1.xlsx")
data_2 = pd.read_excel("../data/data2.xlsx")

# 提取数据
time_periods = data_2['时间段'].dropna().values
average_swimmers = data_2['在池游泳人数平均统计'].dropna().values

# 附件1数据处理
data_1.columns = ['col1', 'col2']
swimmer_counts = data_1['col1'][1:].dropna().values
residual_chlorine = data_1['col2'][1:].dropna().values

print("time_periods: ", time_periods)
print("average_swimmers: ", average_swimmers)
print("swimmer_counts: ", swimmer_counts)
print("residual_chlorine: ", residual_chlorine)


# 使用多项式拟合游泳人数对余氯浓度的影响
def model(x, a, b, c):
    return a * x ** 2 + b * x + c


params, _ = curve_fit(model, swimmer_counts, residual_chlorine)  # 拟合出0.5小时后不同人数与余氯浓度的关系
print("ploy params：", params)

# 模拟从9:00到21:00的余氯浓度变化
init_chlorine = 0.6
cur_chlorine = init_chlorine
chlorine_levels = []  # 余氯浓度
add_chlorine_times = []  # 加入氯的时间

# 每半小时计算一次余氯浓度变化
time_interval = 0.05
data_init_time = 0.5
lst_k = -1000
time_points = np.arange(9, 21, time_interval)

total_cnt = 0  # 总加氯次数
cur_tm = 0
for i, t in enumerate(time_points):
    # print("cur_tm: ", cur_tm)

    cur_avg_people = average_swimmers[int(np.floor(t - 9))]

    cur_people_chlorine = model(cur_avg_people, *params)

    k = -np.log(cur_people_chlorine / init_chlorine) / data_init_time  # 计算出参数 k
    if lst_k == -1000:
        lst_k = k
    if lst_k != k:
        cur_tm = -np.log(cur_chlorine / init_chlorine) / k
        lst_k = k

    # 计算当前时刻的余氯浓度
    cur_time_chlorine = init_chlorine * np.exp(-k * cur_tm)
    cur_chlorine = cur_time_chlorine

    chlorine_levels.append(cur_chlorine)

    if t < 12 < t + time_interval or t < 15 < t + time_interval or t < 18 < t + time_interval:
        if cur_chlorine == init_chlorine:
            continue
        print("cur_chlorine: ", cur_chlorine)
        cur_chlorine = init_chlorine
        total_cnt += 1
        cur_tm = 0
        add_chlorine_times.append(t)
        continue

    if cur_chlorine <= 0.3:
        if 11 <= t < 12 or 14 <= t < 15 or 17 <= t < 18:
            continue
        # print("cur_chlorine: ", cur_chlorine)
        cur_chlorine = init_chlorine
        total_cnt += 1
        cur_tm = 0
        add_chlorine_times.append(t)
        continue

    print(t, cur_chlorine)
    cur_tm += time_interval

chlorine_levels = [round(i, 4) for i in chlorine_levels]
add_chlorine_times = [round(i, 2) for i in add_chlorine_times]

add_std_time = [f'{int(np.floor(i))}:{int(np.round(60 * (i - np.floor(i)), decimals=2))}' for i in add_chlorine_times]
print(time_points)
print(chlorine_levels)

print(add_chlorine_times)
print("再次添加氯的时刻分别为：", add_std_time)

print("总添加次数为：", total_cnt)

# 设置绘图风格
sns.set(style="whitegrid")

# 创建绘图
plt.figure(figsize=(12, 8))
sns.lineplot(x=time_points, y=chlorine_levels, color='#f3b23e', linewidth=2.5)

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
plt.savefig("../img/p3.png")
# 显示图形
plt.show()


