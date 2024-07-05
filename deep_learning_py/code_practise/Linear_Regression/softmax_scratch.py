import torch
from IPython import display  # 从 IPython 导入 display 模块，用于显示功能
from d2l import torch as d2l


class Accumulator:
    """ 一个累加器类，用于在训练过程中累积多个变量的和 """

    def __init__(self, n):
        self.data = [0.0] * n  # 初始化一个长度为 n 的零列表

    def add(self, *args):  # 将传入的多个参数一次累加到数据中
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)  # 重制累加数据，将所有值重制为零

    def __getitem__(self, idx):  # 支持索引访问累加器中的值
        return self.data[idx]


class Animator:
    """ 在动画中绘制数据 """

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量的绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # 使用 lambda 函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):  # 向图表中田间多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)  # 显示当前绘图
        display.clear_output(wait=True)  # 清除输出以便实时更新


def softmax(X):
    """ 计算 X 的 softmax（实现公式） """
    X_exp = torch.exp(X)  # 对 X 进行指数运算（为正数）
    partition = X_exp.sum(1, keepdim=True)  # 按行进行求和，得到区分函数
    return X_exp / partition  # 将每个元素的指数除以区分函数，得到概率分布


def net(X):
    """ 定义神经网络前向传播函数 """
    # X.reshape((-1, W.shape[0])) ———— 改变 X 的形状，使其可以与 W 的行数相匹配，以便进行乘法运算
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)  # 进行矩阵乘法并加上偏置，然后运用 softmax 函数


def cross_entropy(y_hat, y):
    """ 计算交叉熵损失 """
    # y_hat[range(len(y_hat)), y] ———— 按照二维数据访问数据
    # 例如：y_hat[[0, 1], [0, 2]] => 访问 y_hat[0][0], y_hat[1][2] 两个数据,
    # 最终返回 tensor([y_hat[0][0], y_hat[1][2])
    return -torch.log(y_hat[range(len(y_hat)), y])


def updater(batch_size):
    """ 定义更新器函数，使用随机梯度下降算法更新权重 W 和偏置 b"""
    return d2l.sgd([W, b], lr, batch_size)


def accuracy(y_hat, y):
    """ 计算预测正确的数量，计算预测值 y_hat 与真实值 y 的准确率 """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 如果预测值是二维且有多个类别，则取元素值最大值的下标作为预测类别
    cmp = y_hat.type(y.dtype) == y  # 由于 y_hat 的类型可能与 y 不一致，则需要先转换为 y 的类型，然后比较预测值类别与真实值是否相等
    return float(cmp.type(y.dtype).sum())  # 计算预测正确的数量


def evaluate_accuracy(net, data_iter):
    """ 计算在指定数据集上模型的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 如果模型是 nn.Module 的实例，则设置为评估模式（确保在评估时关闭 dropout 和 batch normalization）
    metric = Accumulator(2)  # 初始化累加器，用于存储正确预测的数量和样本总数，以便计算准确率
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源（在评估时不需要计算梯度）
        for X, y in data_iter:
            # net(X) ———— 计算出评测值
            # accuracy(net(X), y) ———— 函数计算预测准确的数量
            # y.numel() ———— 样本的总数
            metric.add(accuracy(net(X), y), y.numel())  # 累加准确预测的数量和总样本数
    return metric[0] / metric[1]  # 返回准确率（分类正确的样本数 / 总样本数）


def train_epoch_ch3(net, train_iter, loss, updater):
    """ 训练一个 epoch，计算损失并反向传播更新模型参数，同时累加损失和准确率"""
    if isinstance(net, torch.nn.Module):  # 设置为训练模式
        net.train()

    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)  # 向前传播得到预测值
        l = loss(y_hat, y)  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):  # 如果 updater 是 torch.optim.Optimizer 的实例或子类的实例
            updater.zero_grad()  # 清空梯度
            l.mean().backward()  # 反向传播
            updater.step()  # 更新参数
        else:
            l.sum().backward()  # 反向传播
            updater(X.shape[0])  # 使用自定义的更新器更新参数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 累加损失和准确率

    # 【总损失】metric[0] 由 float(l.sum()) 累加得到
    # 【总正确预测数】metric[1] 由 accuracy(y_hat, y) 累加得到
    # 【总样本数】metric[2] 由 y.numel() 累加得到
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """ 训练模型若干个 epoch，可视化训练过程并在每个 epoch 后评估测试集的准确率 """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])  # 初始化动画器，用于可视化训练过程
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 训练一个 epoch，返回损失和准确率
        test_acc = evaluate_accuracy(net, test_iter)  # 评估模型在测试集上的准确率
        animator.add(epoch + 1, train_metrics + (test_acc,))  # 更新动画器，绘制当前 epoch 的训练损失
    train_loss, train_acc = train_metrics  # 提取训练损失和训练准确率

    # 这个断言检查 train_loss 是否小于 0.5。如果条件不成立，程序会抛出 AssertionError，并显示 train_loss 的值。
    assert train_loss < 0.5, train_loss
    # 这个断言检查 train_acc 是否介于 0.7 和 1 之间。如果条件不成立，程序会抛出 AssertionError，并显示 train_acc 的值。
    assert 1 >= train_acc > 0.7, train_acc
    # 这个断言检查 test_acc 是否介于 0.7 和 1 之间。如果条件不成立，程序会抛出 AssertionError，并显示 test_acc 的值。
    assert 1 >= test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """ 预测并展示预测结果和真实标签"""
    for X, y in test_iter:
        break  # 只取第一个批次的数据

    # 将数值标签转换为对应的文字标签，存储在 trues 列表中
    trues = d2l.get_fashion_mnist_labels(y)
    # 使用模型 net 对输入数据 X 进行预测，得到的结果通过 argmax(axis=1) 获取每个样本的预测类别，然后转化为对应文字标签，存储在 preds 中
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    # 将真实标签和预测标签结合起来，每个样本生成一个标题，格式为“真实标签\n预测标签”，存储在 titles 列表中
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # 显示前 n 个样本的图像和对应的标题，将 X[0:n] 形状调成为 (n, 28, 28)，每行显示一个图像，并将 titles[0:n] 作为图像的标题
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == "__main__":
    batch_size = 256  # 设置批量大小
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载 Fashion-MNIST 数据集的训练和测试迭代器

    num_inputs = 784  # 输入维度
    num_outputs = 10  # 输出维度

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 初始化权重
    b = torch.zeros(num_outputs, requires_grad=True)  # 初始化偏置
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 创建示例张量
    print("按列计算结果为：", X.sum(0, keepdim=True), "\n按行计算结果为：", X.sum(1, keepdim=True))  # 按行/列求和

    X = torch.normal(0, 1, (2, 5))  # 创建随机张量
    X_prob = softmax(X)  # 计算 softmax
    print("X_prob: ", X_prob, "\nX_prob.sum(1): ", X_prob.sum(1))  # 输出 softmax 结果以及按行求和

    y = torch.tensor([0, 2])  # 创建标签张量
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 创建预测张量
    print(y_hat[[0, 1], y])  # 输出预测值中对应真实值标签的概率

    print(cross_entropy(y_hat, y))  # 输出交叉熵损失

    print(accuracy(y_hat, y) / len(y))  # 输出准确率
    print(evaluate_accuracy(net, test_iter))  # 随机猜测的准确率

    lr = 0.1  # 设置学习率
    num_epochs = 10  # 设置训练周期
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)  # 训练模型

    predict_ch3(net, test_iter)  # 进行预测并展示预测结果
    d2l.plt.show()  # 绘制图形
