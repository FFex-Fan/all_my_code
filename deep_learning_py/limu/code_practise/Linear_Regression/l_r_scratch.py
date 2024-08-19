import random
import torch
from d2l import torch as d2l

""" 表达式： y = Xw + b + 噪声 """
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 均值为 0，方差为 1，维度为 (num_examples, len(w))
    y = torch.matmul(X, w) + b # 计算标签 y
    y += torch.normal(0, 0.01, y.shape) # 添加均值为 0，标准差为 0.01 的噪声
    return X, y.reshape((-1, 1)) # 返回生成的输入数据 X，以及将 y 转化为列向量后返回

""" 批量读取数据，返回特定的特征和标签 """
def data_iter(batch_size, features, labels): 
    num_examples = len(features) # 获取样本个数(1000)
    indices = list(range(num_examples)) # 创建样本索引列表
    random.shuffle(indices) # 随机打乱样本索引，保证样本的随机读取
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
                indices[i : min(i + batch_size, num_examples)] 
        ) # 获取当前批次的样本索引， i + batch_size 的值最大不能超过 num_examples 所以需要取 min
        yield features[batch_indices], labels[batch_indices] # 返回当前批次的特征和标签

""" 定义线性回归模型 """
def linreg(X, w, b):
    return torch.matmul(X, w) + b # 计算预测值：y_hat = Xw + b

""" 定义平方损失函数 """
def squared_loss(y_hat, y, batch_size):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 / batch_size

""" 定义随机梯度下降 """
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad # 更新参数 param
            param.grad.zero_() # 将参数的梯度清零


if __name__ == '__main__':
    #################### 生成数据集 #########################
    true_w = torch.tensor([2, -3.4]) # 定义真实的权重向量 true_w 和 偏置 true_b
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000) # 生成包含 1000 个样本的数据集
    print('features: ', features[0], '\nlabel: ', labels[0]) # 输出第一个样本的特征和标签

    #################### 显示生成的数据集 ###################
    d2l.use_svg_display()
    d2l.set_figsize((8, 8))
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.show()
    print("数据集显示完成！")

    #################### 读取数据集(测试) ###################
    batch_size = 10 # 设置批量大小
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break
    print("数据读取完成！")

    ##################### 初始化模型参数 #####################
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) # 权重向量 w，均值为 0，标准差为 0.01，形状为 (2, 1)，并需要计算梯度
    b = torch.zeros(1, requires_grad=True) # 初始化偏置 b，值为 0，需要计算梯度

    ##################### 训练模型 ##########################
    lr = 0.03 # 学习率
    num_epochs = 3 # 迭代次数
    net = linreg # 模型选择
    loss = squared_loss # 损失函数选择

    for epoch in range(num_epochs): # 开始训练模型
        for X, y in data_iter(batch_size, features, labels): # 遍历所有批次的数据
            l = loss(net(X, w, b), y, batch_size) # 计算预测值和损失（包含每个样本损失的张量）
            # 因为 l 形状是 (batch_size, 1)，而不是一个标量。
            # l 中的所有元素被加到一起，并以此计算关于[w, b] 的梯度
            l.sum().backward() # 求和并计算梯度
            sgd([w, b], lr, batch_size) # 使用梯度下降算法更新参数
        with torch.no_grad(): # 在每个 epoch 结束后，计算并输出训练损失
            train_l = loss(net(features, w, b), labels, batch_size)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b 的估计误差: {true_b - b}')
    print("模型训练完成！")



"""
    1. with torch.no_grad(): 关闭自动求导机制，即：这段代码块中的所有操作都不会被记录在计算图中（在评估模型性能时不用计算梯度）
    2. l.sum().backward() 为什么需要先求和然后再计算梯度？由于 l 是一个张量，通过 l.sum()，确保输入给 backward() 的是一个标量，这样才能正确地计算梯度。
    3. 
"""
