# coding=utf-8
from torch.utils.data import Dataset
import os
from PIL import Image


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(str(self.path))

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        full_path = os.path.join(str(self.path), img_name)
        img = Image.open(full_path)
        return img, self.label_dir  # 设置按下标访问所返回的内容，返回图片及对应的标签

    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    root_dir = "../data/hymenoptera_data/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)

    train_dataset = ants_dataset + bees_dataset