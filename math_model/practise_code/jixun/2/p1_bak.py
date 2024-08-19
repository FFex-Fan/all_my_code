# coding=utf-8
import time

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


df_single = pd.read_csv("data/历史用电数据_单柜.csv")
df_double = pd.read_csv("data/历史用电数据_二并柜.csv")
df_triple = pd.read_csv("data/历史用电数据_三并柜.csv")

def data_processing(df):
    time_data = df[df['system_type'] == 'Time'].reset_index(drop=True)
    new_rows = []
    for date in df["date"].unique():
        time_series = time_data[time_data["date"] == date]["data"].values[0].split(',')
        cn_series = df[(df["date"] == date) & (df["system_type"] == "CN")]["data"].values[0].split(',')
        sd_series = df[(df["date"] == date) & (df["system_type"] == "SD")]["data"].values[0].split(',')

        for i in range(len(time_series)):
            new_rows.append([
                f"{date} {time_series[i] if len(str(time_series[i]).split(':')) == 3 else str(time_series[i] + ':00')}",
                cn_series[i], sd_series[i]])
    new_df = pd.DataFrame(new_rows, columns=["Time", "CN", "SD"])
    filtered_data = new_df[pd.to_datetime(new_df['Time']) > pd.to_datetime('2024-02-20')]
    return filtered_data


df_single = data_processing(df_single)
df_double = data_processing(df_double)
df_triple = data_processing(df_triple)


def calc_load(df):
    df['load'] = df['SD'].astype(float) - df['CN'].astype(float)
    return df


df_single_load = calc_load(df_single)
df_double_load = calc_load(df_double)
df_triple_load = calc_load(df_triple)

out_pd = pd.DataFrame(df_single_load, columns=["Time", "load"])
# out_pd.to_excel("data/output.xlsx", index=False)

print(out_pd)


def train_and_predict_sarima(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 96)):
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.asfreq('15s')
    sarima_model = SARIMAX(df['load'], order=order, seasonal_order=seasonal_order)

    sarima_results = sarima_model.fit(disp=False)

    forecast = sarima_results.get_forecast(steps=len(df_single_load))

    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(seconds=15), periods=len(df_single_load), freq='15S')
    forecast_df = pd.DataFrame({'Time': forecast_index, '预测负荷': forecast.predicted_mean})
    return forecast_df

start_time = time.time()
forecast_single_load = train_and_predict_sarima(df_single_load)
end_time = time.time()
print(forecast_single_load)
print(f"运行时间: {end_time - start_time}秒")
