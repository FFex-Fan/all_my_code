# coding=utf-8
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter(log_dir='logs')
# img_path = '../data/hymenoptera_data/train/ants_img/0013035.jpg'
img_path = '../data/hymenoptera_data/train/bees_img/1232245714_f862fbe385.jpg'

img_PIL = Image.open(img_path)
img_ndarray = np.array(img_PIL)

print(type(img_ndarray))
print(img_ndarray.shape)  # 结果为 (H, W, C) => (高度. 宽度, 通道)

writer.add_image("bees", img_ndarray, 1, dataformats='HWC')

"""
   执行完该语句之后可以使用 tensorboard --logdir=logs 命令查看 tensorboard 
        --logdir=事件文件所在的文件夹名称 (--logdir='test/logs')
        --port=设置端口名称（防止端口冲突，默认为6006）
"""
for i in range(100):
    writer.add_scalar("y = 2x", 3 * i, i)

writer.close()
