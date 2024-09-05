# coding=utf-8
import torch
import torchvision
from model_save import *


# 加载方式一（对应保存方式一）
model = torch.load("checkpoint/vgg16_method1.pth", weights_only=False)
# print(model)

# 加载方式二（对应保存方式二）
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("checkpoint/vgg16_method2.pth", weights_only=False))
# print(vgg16)

# trap，需要导入自定义的网络模型才可以进行模型加载，否则会报错
custom = torch.load("checkpoint/vgg_test_custom_model.pth", weights_only=False)
print(custom)