# coding=utf-8
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32)


# 定义神经网络
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, inp):
        return self.conv1(inp)


net = Net()
stp = 0
writer = SummaryWriter('./logs')
for item in dataloader:
    imgs, labels = item
    outputs = net(imgs)
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("input", imgs, stp)  # torch.Size([32, 3, 32, 32])
    writer.add_images("output", outputs, stp)  # torch.Size([32, 6, 30, 30])
    stp += 1

writer.close()
