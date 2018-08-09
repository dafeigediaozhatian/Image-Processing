import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

%matplotlib inline

## 图像加入标签（整体标签和局部标签）
- MaskRCNN中标签由文件名中数字决定，提取出数字并查询
- PSPnet中由rgb值决定，根据索引查询（rgb中r和b需要换位）
- 图像输入结果为：中心点，整体label，局部label

# MaskRCNN中标签由文件名中数字决定，提取出数字并查询
# 文件名，标签代号
label_num = []

for i in range(len(mask_list[:3])):
    label_num.append(mask_list[i].split(' ')[1:-1])

print(label_num)
label_num[0]

import colorsys
import random

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

for c in range(3):
    mask_color = color[c] * 255

## PSPnet查询rgb值和对应的label
- 根据rgb值得到对应的label
- 将整图的label加入数据中

# 输入路径
name_path = 'label/name'
color_path = 'label/color_rgb'

# 保存路径
save_path = 'label/table'

#　读取路径并建立索引表
list_name = []

with open(color_path, 'r') as f1:
    for i in f1.readlines():
        #print(i.strip().split('\t'))
        list_name.append(i.strip().split('\t'))

with open(name_path, 'r') as f2:
    num = 0
    for i in f2.readlines():
        list_name[num].append(i.strip())
        num += 1
        
#int(list_name[0][0])

for i in list_name:
    #print()
    for j in range(3):
        i[j] = int(i[j])
        
list_name

list_name[11] = ['235', '255', '7', "'sidewalk'"]

list_name

list_name[0][3].split('\'')[1]

for i in list_name:
    i[3] = i[3].split('\'')[1]

list_name

with open(save_path, 'w') as f3:
    for i in list_name:
        #for j in range(len(i)):
        line = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + i[3] + '\n'
            #if j == (len(i)-1):
                #line = line + str(i[j]) + '\n'
            
        #print(line)
        f3.write(line)
