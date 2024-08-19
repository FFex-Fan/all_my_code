# coding=utf-8
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# 设置matplotlib支持中文显示
my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

# 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 划分训练集和测试集, 20% 的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

""" 模型评估
决定系数（R²）是一种衡量模型拟合优度的统计指标，范围在 0 到 1 之间。越接近 1 说明模型的预测结果越好
    计算公式：1 - sum(y - y_hat) ** 2 / sum(y_hat - np.mean(y)) ** 2    
"""
# 打印模型的系数（即每个特征的权重）
print('Coefficients:', model.coef_)
# 打印模型的截距
print('Intercept:', model.intercept_)
# 计算 R² 分数
score = model.score(X_test, y_test)
print('Test score:', score)

# 可视化预测结果与真实值的比较
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='真实值 vs 预测值')
plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='理想情况线')  # 理想情况下预测值等于真实值的线
plt.xlabel('真实值', fontproperties=my_font)
plt.ylabel('预测值', fontproperties=my_font)
plt.title('线性回归模型预测结果', fontproperties=my_font)
plt.legend(prop=my_font)
plt.show()
