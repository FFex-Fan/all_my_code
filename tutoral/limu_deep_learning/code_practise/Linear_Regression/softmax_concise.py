# coding=utf-8
from softmax_scratch import train_ch3
import torch
from torch import nn  # 导入神经网络模块
from d2l import torch as d2l


# 用于初始化网络权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 如果该层是线性层，则使用正态分布初始化权重，标准差为 0.01


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载 Fashion MNIST 数据集，返回训练/测试数据迭代器

    # 定义一个包含展平层和线性层的神将网络
    net = nn.Sequential(
        nn.Flatten(),  # 将输入数据展平成一维
        nn.Linear(784, 10)  # 定义一个线性层，输入特征数为 784（28x28展平），输出特征数为 10（类别数）
    )

    net.apply(init_weights)  # 初始化网络权重

    loss = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数为交叉熵损失

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # 定义优化器为随机梯度下降，学习率为 0.1

    num_epochs = 10  # 设置训练轮数为 10

    # 调用 train_ch3 函数进行训练，传入网络、训练数据迭代器、测试数据迭代器、损失函数、训练轮数和优化器
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    # train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
