# coding=utf-8
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

"""
使用 GPU 需要修改三个位置：
    1. 模型（model）
    2. 损失函数（loss_function）
    3. 数据（data）
"""

# 定义训练的设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if not torch.cuda.is_available() else "cpu")


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# 获取数据集长度（用于计算模型准确率）
len_train_data = len(train_data)
len_test_data = len(test_data)

# 创建数据加载器
dataloader_train = DataLoader(train_data, batch_size=64)
dataloader_test = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
net = Net()
net.to(device)

# 损失函数（交叉熵损失）
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器（随机梯度下降）
lr = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr)

# 模型训练参数
Epoch = 5
train_stp = 0  # 当前训练次数
test_stp = 0

# 使用 tensorboard
writer = SummaryWriter("logs")

start_time = time.time()
for epoch in range(Epoch):
    print(f"Epoch {epoch} ...")

    # 模型开始训练
    net.train()
    for data in dataloader_train:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        cur_out = net(images)

        # 计算损失
        cur_loss = loss_fn(cur_out, labels)

        optimizer.zero_grad()  # 优化器梯度清零
        cur_loss.backward()  # 反向传播
        optimizer.step()  # 进行梯度优化

        train_stp += 1
        if train_stp % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("No: {}, Loss: {}".format(train_stp, cur_loss))
            writer.add_scalar("Loss_train", cur_loss.item(), train_stp)

    # 测试
    net.eval()
    test_loss = 0  # 测试集上的总损失
    test_acc = 0
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            cur_out = net(images)
            cur_loss = loss_fn(cur_out, labels)
            cur_acc = (cur_out.argmax(dim=1) == labels).sum()

            test_loss += cur_loss.item()
            test_acc += cur_acc.item() # 统计总正确个数

    torch.save(net.state_dict(), './checkpoint/epoch_{}.pth'.format(epoch)) # 保存当前训练的模型
    writer.add_scalar("Loss_test", test_loss, test_stp)
    writer.add_scalar("Acc_test", test_acc / len_test_data, test_stp)
    test_stp += 1

    print(f"Total loss in test_dataset: {test_loss}")
    print(f"Total accuracy in test_dataset: {test_loss / len_test_data}")
    print("saved")

writer.close()