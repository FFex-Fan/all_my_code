# coding=utf-8
import os

root_dir = '../data/hymenoptera_data/train'
target_dir = 'bees_img'
out_dir = 'bees_label'

img_list = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
os.mkdir(os.path.join(root_dir, out_dir))

for i in img_list:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)
