import numpy as np
import torch
from torch.utils import data # 小批量处理数据
from d2l import torch as d2l
from torch import nn

""" 
	构造PyTorch数据迭代器

	参数:
	- data_arrays (tuple): 包含特征和标签的元组 (features, labels)
	- batch_size (int): 每个批次的数据量
	- is_train (bool): 是否打乱数据，默认为 True

	返回:
	- DataLoader: 返回数据迭代器
"""
def load_array(data_arrays, batch_size, is_train=True):
	dataset = data.TensorDataset(*data_arrays) # 创建一个 TensorDataset，将特征和标签打包成一个数据集
	return data.DataLoader(dataset, batch_size, shuffle=is_train) # 创建一个 DataLoader，从数据集中加载数据


if __name__ == '__main__':
	########### 生成数据集
	true_w = torch.tensor([2, -3.4]) # 生成用于合成数据的真实权重
	true_b = 4.2 # 生成用于合成数据的真实偏置
	features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 使用权重和偏置生成 1000 个样本的合成数据

	########### 读取数据集
	batch_size = 10 # 设置批量的大小
	data_iter = load_array((features, labels), batch_size) # 为数据集创建迭代器
	next(iter(data_iter)) # 从数据迭代器中获取第一个批量（测试）

	########### 定义模型
	net = nn.Sequential(nn.Linear(2, 1)) # 创建一个具有 2 个输入和 1 个输出的简单线性神经网络

	########### 模型初始化参数
	net[0].weight.data.normal_(0, 0.01) # 使用正态分布（均值为 0，标准差为 0.01）初始化权重
	net[0].bias.data.fill_(0) # 将偏置初始化为 0

	########### 定义损失函数
	loss = nn.MSELoss() # 使用均方误差损失函数

	########### 定义优化算法
	trainer = torch.optim.SGD(net.parameters(), lr=0.03) # 使用学习率为 0.03 的随机梯度下降优化器

	########### 训练模型 
	num_epochs = 3 # 定义训练的轮数
	for epoch in range(num_epochs): 
		for X, y in data_iter: # 遍历每个数据迭代器中的每个批量
			l = loss(net(X), y) # 计算一个批量损失
			trainer.zero_grad() # 清除以前的梯度
			l.backward() # 反向传播计算梯度
			trainer.step() # 使用梯度更新模型参数
		l = loss(net(features), labels) # 计算整个数据集的损失
		print(f'epoch {epoch + 1}, loss {l:f}') # 打印当前的轮数和损失

	w = net[0].weight.data # 获取训练后的权重
	print('w 的估计误差： ', true_w - w.reshape(true_w.shape)) # 打印权重的估计误差
	b = net[0].bias.data # 获取训练后的偏置
	print('b 的估计误差： ', true_b - b) # 打印偏置的估计误差


"""
	1. 为数据创建迭代器的具体过程细节实现?
"""









