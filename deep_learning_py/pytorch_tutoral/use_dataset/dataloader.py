# coding=utf-8
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 获取数据集并将 PIL 类型的数据转换为 tensor 类型
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# 创建数据加载器，分批次的输出数据
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2): # shuffle=True 时两次结果不同
    stp = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, stp)
        stp += 1

writer.close()