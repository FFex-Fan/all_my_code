# coding=utf-8
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

boston = fetch_california_housing()
x = boston.data # 特征数据
y = boston.target # 目标值（房价）

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 标准化训练集
X_test = scaler.transform(X_test) # 标准化测试集

# 转化为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 定义神经网络模型
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )

    def forward(self, inp):
        return self.model(inp)

# 实例化模型
model = HousingModel()

# 损失函数
loss_fn = nn.MSELoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 100

model.train() # 训练模式
for epoch in range(epochs):
    optimizer.zero_grad() # 梯度清零
    output = model(X_train) # 前向传播
    loss = loss_fn(output, y_train) # 计算损失
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

    if epoch % 10 == 0:
        print(f'Epoch {epoch} / {epochs}, Loss: {loss.item():.4f}')

# 模型评估
model.eval() # 评估模式
with torch.no_grad(): # 评估期间不计算梯度
    predictions = model(X_test) # 进行预测
    test_loss = loss_fn(predictions, y_test) # 计算测试的损失
    print(f'Test Loss: {test_loss.item():.4f}')

# 打印前5个预测值与实际值
for i in range(5):
    print(f'Predicted: {predictions[i].item():.4f}, Actual: {y_test[i].item():.4f}')





