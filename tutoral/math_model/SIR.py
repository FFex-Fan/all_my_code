# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import font_manager



"""
scipy.integrate.odeint(func, y0, t, args=())
    - func：定义微分方程的函数。它应该返回一个包含导数的列表。
    - y0：初始条件，表示在时间 t=0 时系统的状态。
    - t：时间点的数组，表示我们希望在这些时间点上求解ODE。
    - args：传递给func的其他参数，以元组的形式给出
"""
def sir_model(y, t, beta, gamma): # 定义 SIR 模型的微分方程
    S, I, R = y
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]


N = 1000
I0 = 10
S0 = N - I0
R0 = 0
beta = 0.2
gamma = 0.1
y0 = [S0, I0, R0]

# 时间点
t = np.linspace(0, 160, 160)

# 使用 odeint 求解
solution = sp.integrate.odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = solution.T

my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='易感者(S)')
plt.plot(t, I, label='感染者(I)')
plt.plot(t, R, label='康复者(R)')
plt.xlabel('时间', fontproperties=my_font)
plt.ylabel('人数', fontproperties=my_font)
plt.legend(prop=my_font)
plt.title('SIT 模型', fontproperties=my_font)
plt.show()
