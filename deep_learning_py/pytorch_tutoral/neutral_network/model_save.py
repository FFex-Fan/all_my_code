# coding=utf-8
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)


# 保存方式一，保存内容为：模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式二，保存内容为：模型参数（推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 方式一 trap
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)

net = Net()
torch.save(net, "vgg_test_custom_model.pth")