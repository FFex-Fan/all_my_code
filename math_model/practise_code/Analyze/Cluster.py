import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 生成模拟数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
# 生成三个簇的二维数据，每个簇包含100个数据点
X = np.vstack((np.random.randn(100, 2) + [2, 2],
               np.random.randn(100, 2) + [-2, -2],
               np.random.randn(100, 2) + [2, -2]))

# 将数据转化为pandas DataFrame
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])  # 创建包含两列（特征）的DataFrame

# 标准化数据
scaler = StandardScaler()  # 创建StandardScaler对象
X_scaled = scaler.fit_transform(df)  # 对数据进行标准化，使每个特征的均值为0，标准差为1

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)  # 创建KMeans对象，设定聚类数为3
kmeans.fit(X_scaled)  # 使用标准化后的数据进行聚类

# 获取聚类结果
labels = kmeans.labels_  # 获取每个数据点的簇标签
centroids = kmeans.cluster_centers_  # 获取每个簇的中心点

# 评估聚类效果
silhouette_avg = silhouette_score(X_scaled, labels)  # 计算轮廓系数（Silhouette Score）
print(f'Silhouette Score: {silhouette_avg}')  # 打印轮廓系数，值越大表示聚类效果越好

# 可视化聚类结果
plt.figure(figsize=(10, 6))  # 创建一个图表，设置尺寸为10x6
# 绘制每个数据点，用不同颜色表示不同簇
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')
# 绘制每个簇的中心点，颜色为红色，标记为'x'
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-means Clustering')  # 设置图表标题
plt.xlabel('Feature1')  # 设置x轴标签
plt.ylabel('Feature2')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图表