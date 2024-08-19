# coding=utf-8
import pandas as pd

file_path = 'data/历史用电数据_三并柜.csv'
historical_data_three = pd.read_csv(file_path)

time_data_three = historical_data_three[historical_data_three['system_id'] == -1]['data'].iloc[0].split(',')
cn_data_three = historical_data_three[historical_data_three['system_id'] == 1]['data'].iloc[0].split(',')
sd_data_three = historical_data_three[historical_data_three['system_id'] == 4]['data'].iloc[0].split(',')

time_data_three = ['00:00:00' if t == '00:00' else t for t in time_data_three]  # 修正第一个时间点
time_data_three = [t if len(t.split(':')) == 3 else f"{t}:00" for t in time_data_three]  # 补全秒
time_data_three = pd.to_timedelta(time_data_three + ['24:00:00'])  # 添加 '24:00:00' 以匹配一天的结束

time_data_three = pd.to_datetime(time_data_three.total_seconds(), unit='s').time

cn_data_three = pd.to_numeric(cn_data_three)
sd_data_three = pd.to_numeric(sd_data_three)

cn_data_three = pd.Series(cn_data_three, index=time_data_three[:-1])
sd_data_three = pd.Series(sd_data_three, index=time_data_three[:-1])

load_data_three = sd_data_three - cn_data_three

full_time_index_three = pd.date_range("00:00:00", "23:59:45", freq="15s").time

# 执行线性插值并填充缺失值
interpolated_load_three = load_data_three.reindex(full_time_index_three).interpolate(method='linear').bfill().ffill()

interpolated_load_df_three = interpolated_load_three.reset_index()
interpolated_load_df_three.columns = ['Time', 'Load']

# 保存为Excel文件
output_file_path_three = 'data/p1_三并柜.xlsx'
interpolated_load_df_three.to_excel(output_file_path_three, index=False)

