# coding=utf-8
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("../imgs/1_LLVL8xUiUOBE8WHgzAuY-Q.png")
# img = Image.open("../imgs/logo.PNG")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize ---- 修改模型参数，使其呈现出不同的状态
"""
    归一化计算公式：input[channel] = (input[channel] - mean[channel]) / std[channel]
    例如：
        参数: mean=[0.5, 0.5], std=[0.5, 0.5], input=[0, 1]
        计算: ([0, 1] - [0.5, 0.5]) / 0.5 = 2 * [0, 1] - 1 => result = [-1, 1]
"""
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)  # input=img_tensor
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> Resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_tensor tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2 (将上述的 resize & totensor 功能进行合并)
trans_resize_2 = transforms.Resize(512)  # 整体缩放
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 传入的参数为一个列表, 列表中要注意位置顺序
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop((500, 1000))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
