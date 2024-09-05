# coding=utf-8
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 进行tensor的转换
])

train_set = torchvision.datasets.CIFAR10(root='../data', train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, transform=dataset_transform, download=False)

# print(test_set[0])
# print(test_set.classes)

writer = SummaryWriter("p10") # 指定日志存放的文件夹名称
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()