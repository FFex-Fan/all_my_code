# coding=utf-8
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# inp = torch.tensor([[1, -0.5],
#                     [-1, 3]])

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class NetSigmoid(nn.Module):
    def __init__(self):
        super(NetSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class NetRelu(nn.Module):
    def __init__(self):
        super(NetRelu, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


sigmoid = NetSigmoid()
relu = NetRelu()
writer = SummaryWriter("./logs")
stp = 0

for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, stp)
    outputs = relu(imgs)
    writer.add_images("output", outputs, stp)
    stp += 1
# print(relu(inp))

writer.close()
