# coding=utf-8
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                  transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


# inp = torch.tensor([[1, 2, 0, 3, 1],
#                     [0, 1, 2, 3, 1],
#                     [1, 2, 1, 0, 0],
#                     [5, 2, 3, 1, 1],
#                     [2, 1, 0, 1, 1]]).reshape((1, 1, 5, 5))

# 搭建网络
class MaxPoolNet(nn.Module):
    def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, inp):
        return self.maxpool(inp)


net = MaxPoolNet()
# out = net(inp)
# print(out)

writer = SummaryWriter(log_dir='./logs')
stp = 0

for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, global_step=stp)
    outputs = net(imgs)
    writer.add_images("output", outputs, global_step=stp)
    stp += 1

writer.close()
