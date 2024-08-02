# coding=utf-8

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)

    def forward(self, x):
        return self.linear1(x)


net = Net()

for data in dataloader:
    imgs, labels = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = net(output)
    print(output.shape)
