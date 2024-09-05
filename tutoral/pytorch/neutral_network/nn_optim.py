# coding=utf-8
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model1(x)


loss = nn.CrossEntropyLoss()
net = Net()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0.0
    for data in dataloader:
        imgs, labels = data
        outputs = net(imgs)
        cur_loss = loss(outputs, labels)
        optim.zero_grad()
        cur_loss.backward()
        optim.step()
        total_loss += cur_loss.item()
    print(f"Epoch: {epoch}, loss_calc: {total_loss}")