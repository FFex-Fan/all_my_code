# coding=utf-8
import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    """ 定义 ReLU 激活函数 """
    a = torch.zeros_like(X)  # 创建一个形状与 X 完全相同的零张量
    return torch.max(X, a)  # 返回 X 和零张量逐元素的最大值


def net(X):
    """ 定义神经网络 """
    X = X.reshape((-1, num_inputs))  # 将输入展平成二维张量，-1 表示自动计算行数
    H = relu(X @ W1 + b1)  # 计算第一个隐藏层的输出，并使用激活函数
    return (H @ W2 + b2)  # 计算输出层的结果


batch_size = 256  # 设置批量大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载 Fashion MNIST 数据集

# 输入层 784 个节点（28x28像素），输出层 10 个节点（10 类），隐藏层 256 个节点
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)  # 初始化第一个全连接层的权重
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 初始化第一个全连接层的偏置
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]  # 将所有参数收集到一个列表中，便于后续的优化

loss = nn.CrossEntropyLoss(reduction='none')  # 使用交叉熵损失函数

num_epochs, lr = 10, 0.1  # 10 个训练周期，学习率为 0.1
updater = torch.optim.SGD(params, lr) # 使用随机梯度下降进行优化

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater) # 训练模型

d2l.predict_ch3(net, test_iter) # 在测试集上预测模型

d2l.plt.show()
