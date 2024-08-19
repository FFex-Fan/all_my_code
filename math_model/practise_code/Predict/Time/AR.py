import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 构建和拟合AR模型（这里使用ARIMA模型，设置差分阶数d=0）
from statsmodels.tsa.arima.model import ARIMA

# 生成模拟的时间序列数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
n = 100  # 数据点的数量
e = np.random.normal(size=n)  # 生成服从正态分布的随机误差
X = np.zeros(n)  # 初始化时间序列数组
for t in range(2, n):  # 从第三个数据点开始生成时间序列数据
    X[t] = 0.6 * X[t - 1] + 0.2 * X[t - 2] + e[t]  # 使用AR(2)模型生成数据

# 将数据转化为pandas Series
time_series = pd.Series(X)  # 将numpy数组转换为pandas Series对象，便于后续处理

# 绘制时间序列图
time_series.plot(title='Simulated Time Series Data')  # 绘制时间序列数据图
plt.xlabel('Time')  # 设置x轴标签
plt.ylabel('Value')  # 设置y轴标签
plt.show()  # 显示图表

# 设置AR模型的阶数p
p = 2  # 设置AR模型的滞后阶数为2
model = ARIMA(time_series, order=(p, 0, 0))  # 构建ARIMA模型，这里差分阶数d设置为0，仅使用自回归部分
result = model.fit()  # 拟合模型

# 打印模型参数
print('Model Parameters:')
print(result.params)  # 打印模型的估计参数

# 检查模型残差
residuals = result.resid  # 获取模型的残差
plt.figure(figsize=(10, 6))  # 创建图表并设置尺寸
plt.plot(residuals)  # 绘制残差图
plt.title('Residuals of the AR Model')  # 设置图表标题
plt.xlabel('Time')  # 设置x轴标签
plt.ylabel('Residuals')  # 设置y轴标签
plt.show()  # 显示图表

# 残差的自相关图
sm.graphics.tsa.plot_acf(residuals, lags=20)  # 绘制残差的自相关图，滞后阶数为20
plt.show()  # 显示图表

# 预测未来10个时刻的值
forecast = result.get_forecast(steps=10)  # 进行未来10个时刻的预测
forecast_values = forecast.predicted_mean  # 获取预测值
print('Forecasted Values:')
print(forecast_values)  # 打印预测值

# 绘制实际值和预测值
plt.figure(figsize=(10, 6))  # 创建图表并设置尺寸
plt.plot(time_series, label='Actual Data')  # 绘制实际时间序列数据
plt.plot(np.arange(len(time_series), len(time_series) + len(forecast_values)), forecast_values, label='Forecasted Data',
         color='red')  # 绘制预测值，颜色为红色
plt.legend()  # 显示图例
plt.title('Actual vs Forecasted Data')  # 设置图表标题
plt.xlabel('Time')  # 设置x轴标签
plt.ylabel('Value')  # 设置y轴标签
plt.show()  # 显示图表