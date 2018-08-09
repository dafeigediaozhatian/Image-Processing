import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

%matplotlib inline

#mask_read_path = 'Mask_RCNN'
#PSPNet = 'Orinigal_image'
PSPNet = 'Image_Fusion'
#mask_save_path = 'Mask_image'
image_save_path = 'Canny_edge'

#read_list = os.listdir(mask_read_path)
#read_list.sort()
PSP_list  = os.listdir(PSPNet)
PSP_list.sort()

#PSP_list = PSP_list[:5]

for i in range(len(PSP_list)):
    image_path = os.path.join(PSPNet, PSP_list[i])
    image = cv2.imread(image_path, 1)
    edges = cv2.Canny(image,100,200)

    # 保存到对应路径
    save_path = os.path.join(image_save_path, PSP_list[i])
    cv2.imwrite(save_path, edges)
    if i % 100 == 0:
        #print("the %d image named %s has been saved" % (i, image_list[i]))
        print("%d images has been saved" % (i))
    if i == 999:
        print("process over")
