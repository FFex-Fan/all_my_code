"""多层感知机简易实现"""
import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 如果是线性层，则使用标准差为 0.01 的正态分布初始化权重


if __name__ == '__main__':
    net = nn.Sequential(
        nn.Flatten(),  # 将输入展平
        nn.Linear(784, 256),  # 输入层到隐藏层的全联接层（784 -> 256）
        nn.ReLU(),  # 隐藏层使用 ReLU 激活函数
        nn.Linear(256, 10)  # 从隐藏层到输出层的全联接层（256 -> 10）
    )
    net.apply(init_weights)  # 应用初始化函数初始化网络的权重

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')  # 使用交叉熵损失函数
    trainer = torch.optim.SGD(net.parameters(), lr)  # 使用随机梯度下降进行优化

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载数据集

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 训练模型
    d2l.plt.show()
