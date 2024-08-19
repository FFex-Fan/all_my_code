import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")

file_path = 'data/weather_data.xlsx'
weather_data = pd.read_excel(file_path)

weather_data['Unnamed: 0'] = weather_data['Unnamed: 0'].str.strip()

weather_data['Time'] = pd.to_datetime(weather_data['Unnamed: 0'], format='%H:%M:%S').dt.time

date_range = pd.date_range(start="2024-09-08", end="2024-09-10 23:00:00", freq='h')
weather_data['DateTime'] = date_range

weather_data['DateTime_Simplified'] = weather_data['DateTime'].dt.strftime('%m-%d %H:%M')

predict_temp = []


def adjust_temperature(temp):
    if temp >= 35:
        tmp = temp - np.random.uniform(3.5, 4)
        predict_temp.append(tmp)
        return tmp
    else:
        tmp = temp - np.random.uniform(1.5, 2.5)
        predict_temp.append(tmp)
        return tmp


weather_data['池水温度'] = weather_data['in_temperature_2024'].apply(adjust_temperature)

sns.color_palette('pastel')
plt.figure(figsize=(14, 7))
sns.lineplot(data=weather_data, x='DateTime_Simplified', y='池水温度',
             label='池水温度', color="#ffc75f", errorbar=('ci', 95))
plt.title('2024 年 9 月 8 日 - 10 日的杭电游泳馆池水温度的变化曲线', fontproperties=my_font)
plt.xlabel('DateTime')
plt.ylabel('温度 (°C)', fontproperties=my_font)
plt.legend(prop=my_font)
plt.grid(True)
plt.xticks(rotation=45)
plt.xticks(np.arange(0, len(weather_data), 6), labels=weather_data['DateTime_Simplified'][::6], rotation=45)

plt.savefig('../img/p4_1.png')
plt.savefig('../img/p4_1.svg')
plt.show()

weather_data['Date'] = weather_data['DateTime'].dt.date
daily_max_temp = weather_data.groupby('Date')['池水温度'].max()
daily_min_temp = weather_data.groupby('Date')['池水温度'].min()

specific_times = ['10:00:00', '12:00:00', '14:00:00', '16:00:00', '20:00:00']
specific_temp = weather_data[weather_data['Time'].astype(str).isin(specific_times)]

print("最高温如下：", daily_max_temp, "\n最低温如下：", daily_min_temp)
for idx, i in enumerate(predict_temp):
    if idx % 24 == 0:
        print(f"\n9 月 {8 + idx // 24} 日 10、12、14、16、20点的池水温度：")
    if idx in [tmp + 24 * t for t in range(0, 3) for tmp in [10, 12, 14, 16, 20]]:
        print(predict_temp[idx])


# 创建一个示例DataFrame
data = {
    '9_8': [i for i in predict_temp[:24]],
    '9_9': [i for i in predict_temp[24:48]],
    '9_10': [i for i in predict_temp[48:]]
}
df = pd.DataFrame(data)

# 将DataFrame写入Excel文件
df.to_excel('../data/output.xlsx', index=False)
