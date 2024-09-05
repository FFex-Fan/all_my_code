# coding=utf-8
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

""" 通过 transforms.ToTensor 解决两个问题
        1. transforms 如何使用
        2. 为什么需要 tensor 数据类型
"""
if __name__ == '__main__':
    img_path = "../data/hymenoptera_data/train/ants_img/0013035.jpg"
    img = Image.open(img_path)

    writer = SummaryWriter('logs')

    # 1. transforms 如何使用
    tensor_trans = transforms.ToTensor()  # 可以理解为创建一个具体的工具
    tensor_img = tensor_trans(img)  # 使用该工具 -> 将 img 作为该工具的输入

    # 2. 为什么需要 tensor 数据类型



    writer.add_image("Tensor_img", tensor_img)
    writer.close()
