import pandas as pd

# 加载历史数据
file_path = 'data/历史用电数据_单柜.csv'
historical_data = pd.read_csv(file_path)

# 提取CN（储能数据）和SD（市电数据）的相关数据
time_data = historical_data[historical_data['system_id'] == -1]['data'].iloc[0].split(',')
cn_data = historical_data[historical_data['system_id'] == 1]['data'].iloc[0].split(',')
sd_data = historical_data[historical_data['system_id'] == 4]['data'].iloc[0].split(',')

time_data = ['00:00:00' if t == '00:00' else t for t in time_data]  # 修正第一个时间点
time_data = [t if len(t.split(':')) == 3 else f"{t}:00" for t in time_data]  # 补全秒
time_data = pd.to_timedelta(time_data + ['24:00:00'])  # 添加 '24:00:00' 以匹配一天的结束

time_data = pd.to_datetime(time_data.total_seconds(), unit='s').time

cn_data = pd.to_numeric(cn_data)
sd_data = pd.to_numeric(sd_data)

cn_data = pd.Series(cn_data, index=time_data[:-1])
sd_data = pd.Series(sd_data, index=time_data[:-1])

# 计算负荷 = 市电 - 储能
load_data = sd_data - cn_data

# 生成从00:00:00到23:59:45每15秒的时间序列
full_time_index = pd.date_range("00:00:00", "23:59:45", freq="15s").time

# 执行线性插值并填充缺失值
interpolated_load = load_data.reindex(full_time_index).interpolate(method='linear').bfill().ffill()

interpolated_load_df = interpolated_load.reset_index()
interpolated_load_df.columns = ['Time', 'Load']
output_file_path = 'data/p1_单柜.xlsx'
interpolated_load_df.to_excel(output_file_path, index=False)

