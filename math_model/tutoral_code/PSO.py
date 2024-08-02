import numpy as np
import matplotlib.pyplot as plt


# 定义Schaffer函数
def PSO_Schaffer(x1, x2):
    temp1 = x1 ** 2 + x2 ** 2
    temp2 = np.sin(np.sqrt(temp1))
    return 0.5 + (temp2 ** 2 - 0.5) / (1 + 0.001 * temp1) ** 2


# 定义变量范围和步长
x1 = np.arange(-10, 10.05, 0.05)  # 生成从 -10 到 10，步长为 0.05 的数组
x2 = np.arange(-10, 10.05, 0.05)  # 同上
X1, X2 = np.meshgrid(x1, x2)  # 创建网格

# 计算网格上每个点的Schaffer函数值
f = PSO_Schaffer(X1, X2)

# 创建三维曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, f, cmap='viridis')  # 绘制三维曲面图
plt.title('Schaffer function')
plt.show()

# 粒子群算法初始化参数
w = 0.5  # 惯性权重
c1 = 1  # 个体学习因子
c2 = 1  # 社会学习因子
maxg = 200  # 最大迭代次数
N = 100  # 粒子数

# 粒子位置和速度的范围
Vmax = 1
Vmin = -1
Xmax = 10
Xmin = -10
dim = 2  # 问题维度

# 初始化N个粒子的位置和速度
location = Xmax * (2 * np.random.rand(N, dim) - 1)  # 随机初始化位置在 [-Xmax, Xmax] 之间
V = Vmax * (2 * np.random.rand(N, dim) - 1)  # 随机初始化速度在 [-Vmax, Vmax] 之间
fitness = np.array([PSO_Schaffer(location[i, 0], location[i, 1]) for i in range(N)])  # 计算初始适应度

# 初始化个体最优解和群体最优解
fitnessgbest = np.min(fitness)  # 全局最优适应度
bestindex = np.argmin(fitness)  # 全局最优适应度对应的粒子索引
gbest = location[bestindex, :]  # 全局最优位置
pbest = location.copy()  # 个体最优位置
fitnesspbest = fitness.copy()  # 个体最优适应度

# 存储每次迭代的全局最优适应度
yy = np.zeros(maxg)

# 迭代寻优
for i in range(maxg):
    for j in range(N):
        # 更新速度
        V[j, :] = w * V[j, :] + c1 * np.random.rand() * (pbest[j, :] - location[j, :]) + c2 * np.random.rand() * (
                    gbest - location[j, :])
        V[j, :] = np.clip(V[j, :], Vmin, Vmax)  # 限制速度在[Vmin, Vmax]之间

        # 更新位置
        location[j, :] = location[j, :] + V[j, :]
        location[j, :] = np.clip(location[j, :], Xmin, Xmax)  # 限制位置在[Xmin, Xmax]之间

        # 更新适应度
        fitness[j] = PSO_Schaffer(location[j, 0], location[j, 1])

        # 更新个体最优解
        if fitnesspbest[j] > fitness[j]:
            pbest[j, :] = location[j, :]
            fitnesspbest[j] = fitness[j]

        # 更新全局最优解
        if fitnessgbest > fitness[j]:
            gbest = location[j, :]
            fitnessgbest = fitness[j]

    yy[i] = fitnessgbest  # 存储每次迭代的全局最优适应度

# 绘制群体最优适应度随迭代次数的变化图
plt.figure()
plt.plot(yy)
plt.title('Best Optinal', fontsize=12)
plt.xlabel('Epoch num', fontsize=18)
plt.ylabel('fit degree', fontsize=18)
plt.show()
