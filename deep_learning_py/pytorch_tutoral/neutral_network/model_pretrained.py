import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

""" 
    当前结果分为 1000 类，若需要将结果分为 10 类，则有如下方法：
"""

# method 1
# vgg16_true.add_module('add_linear', nn.Linear(1000, 10)) # 将该线性层添加到 VGG 中（与 classifier 同级）
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10)) # 将该线性层添加到 classifier 中（属于 classifier）
print(vgg16_true)

# method 2
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(1000, 10) # 直接修改 classifier[6] 层对应的模型
print(vgg16_false)