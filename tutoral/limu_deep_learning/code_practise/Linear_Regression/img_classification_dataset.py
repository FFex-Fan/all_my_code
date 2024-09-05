import torch
import torchvision # 导入计算机视觉包
from torch.utils import data # 导入数据处理模块
from torchvision import transforms # 从 torchvision 导入图像变换模块
from d2l import torch as d2l

""" 返回 Fashion-MNIST 数据集的文本标签 """
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt' 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker',' bag', 'ankle boot']  # 定义 Fashion-MNIST 数据集的标签名称
    return [text_labels[int(i)] for i in labels]  # 将数字标签转为对应的文字标签并返回


""" 绘制图像列表 """
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)  # 设置需要显示图像的尺寸
    
    # 创建一个指定大小的图像网格，figsize 决定每个子图的大小
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() # 将 2D 网格子图展平 1D 数组，方便后续的遍历

    for i, (ax, img) in enumerate(zip(axes, imgs)): # 遍历所有图像及其对应的坐标轴（enumerate 自带 index）
        if torch.is_tensor(img): 
            ax.imshow(img.numpy()) # 如果图像是张量，则转换为 numpy 数组显示
        else: 
            ax.imshow(img)# 否则直接显示 PIL 图片
        
        ax.axes.get_xaxis().set_visible(False) # 隐藏 x 轴
        ax.axes.get_yaxis().set_visible(False) # 隐藏 y 轴
        
        if titles:
            ax.set_title(titles[i]) # 如果提供了标题，则设置标题
    return axes # 返回所有的坐标轴对象


""" 使用 4 个进程来读取数据"""
def get_dataloader_workers():
    return 4  # 返回数据加载器使用的工作进程数，以加速数据加载过程


""" 下载 Fashion-MNIST 数据集，然后将其家在到内存中 """
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()] # 定义基本的图像变换（转换为张量）
    
    if resize:
        trans.insert(0, transforms.Resize(resize)) # 如果给定了尺寸大小，则调整尺寸的变换
     
    # 将所有变换组合成一个复合变换
    trans = transforms.Compose(trans)
    
    # 下载并加载训练/测试数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True) 
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    
    # 返回训练/测试数据集的迭代器
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    


if __name__ == "__main__":
    d2l.use_svg_display() # 使用 svg 格式显示图片（矢量图）

    trans = transforms.ToTensor() # 定义基本的图像变换（转换为张量）
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True) # 下载并加载训练数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True) # 下载并加载测试数据集

    print("mnist_train 的大小为：", len(mnist_train), "\nmnist_test 的大小为：", len(mnist_test))
    print("每个图像的形状为：", mnist_train[0][0].shape)
    
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18))) # 从训练数据集中获取第一批数据
    
    # 按 2 行 9 列的布局，绘制这批数据的图像
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y)) 
    # d2l.plt.show() # 展示出所绘制的图像
    
    batch_size = 256 # 设置批量大小为 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, # 创建训练数据的迭代器
                                 num_workers=get_dataloader_workers()) # 使用 4 个进程并行加载数据
    
    timer = d2l.Timer() # 创建计时器对象
    for X, y in train_iter: # 遍历训练数据，什么都不做，只为了计时
        continue
    print(f'{timer.stop():.2f} sec') # 输出遍历训练数据所需的时间
    
    # 加载调整大小后的数据集，批次大小为 32， 图像尺寸为 64 x 64
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, ' --------- ', y.shape, y.dtype) # 输出数据批次的形状和数据类型
        break # 输出完第一批数据后就退出
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    