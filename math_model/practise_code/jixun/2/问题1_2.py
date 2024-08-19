# coding=utf-8
import pandas as pd

# 加载二并柜的历史用电数据
file_path = 'data/历史用电数据_二并柜.csv'
historical_data_two = pd.read_csv(file_path)

# 提取二并柜的CN（储能数据）和SD（市电数据）的相关数据
time_data_two = historical_data_two[historical_data_two['system_id'] == -1]['data'].iloc[0].split(',')
cn_data_two = historical_data_two[historical_data_two['system_id'] == 1]['data'].iloc[0].split(',')
sd_data_two = historical_data_two[historical_data_two['system_id'] == 4]['data'].iloc[0].split(',')

# 标准化时间格式为 hh:mm:ss
time_data_two = ['00:00:00' if t == '00:00' else t for t in time_data_two]  # 修正第一个时间点
time_data_two = [t if len(t.split(':')) == 3 else f"{t}:00" for t in time_data_two]  # 补全秒
time_data_two = pd.to_timedelta(time_data_two + ['24:00:00'])  # 添加 '24:00:00' 以匹配一天的结束


time_data_two = pd.to_datetime(time_data_two.total_seconds(), unit='s').time

cn_data_two = pd.to_numeric(cn_data_two)
sd_data_two = pd.to_numeric(sd_data_two)

# 使用时间作为索引创建CN和SD的Series
cn_data_two = pd.Series(cn_data_two, index=time_data_two[:-1])
sd_data_two = pd.Series(sd_data_two, index=time_data_two[:-1])

# 计算负荷 = 市电 - 储能
load_data_two = sd_data_two - cn_data_two

# 生成从00:00:00到23:59:45每15秒的时间序列
full_time_index_two = pd.date_range("00:00:00", "23:59:45", freq="15s").time

# 执行线性插值并填充缺失值
interpolated_load_two = load_data_two.reindex(full_time_index_two).interpolate(method='linear').bfill().ffill()

interpolated_load_df_two = interpolated_load_two.reset_index()
interpolated_load_df_two.columns = ['Time', 'Load']

# 保存为Excel文件
output_file_path_two = 'data/p1_二并柜.xlsx'
interpolated_load_df_two.to_excel(output_file_path_two, index=False)

