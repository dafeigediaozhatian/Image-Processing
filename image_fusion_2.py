import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

%matplotlib inline

# # 读取原图和mask文件
filename1 = 'PSPNet/color/591.png'
filename3 = '1736.jpg'

psp = cv2.imread(filename3, 1)

plt.hist(psp.ravel(), 256, [0,256]);
plt.show()

# 读取原图和mask文件
filename1 = 'PSPNet/color/591.png'
filename2 = 'Mask RCNN/1736.jpg [57 57 57 57 57 57 57 57] .npy'
filename3 = '1736.jpg'

PSP = cv2.imread(filename1, 1)
Mask = np.load(filename2)
image = cv2.imread(filename3, 1)

# 将mask矩阵中加入值，便于进行位运算
Mask = Mask + 0
Mask.dtype = 'uint8'

# 将mask加入原图中，得到效果图
mask_image_1 = Mask[:, :, 0]
#mask_image_2 = Mask[:, :, 1]
#mask_image_3 = Mask[:, :, 2]

#mask_image = cv2.merge(mask_image_1, mask_image_2, mask_image_3)

#print(mask_image_1.dtype)

#ret, mask = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)

#Mask = np.load(filename2)
#Mask[:, :, 0]

#mask_inv = cv2.bitwise_not(mask_image_1)

img2_fg = cv2.bitwise_and(image,image,mask = mask_image_1)

cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 读取原图和mask文件
filename1 = 'PSPNet/color/591.png'
filename2 = 'Mask RCNN/1736.jpg [57 57 57 57 57 57 57 57] .npy'
filename3 = '1736.jpg'

PSP = cv2.imread(filename1, 1)
Mask = np.load(filename2)
image = cv2.imread(filename3, 1)

# 将mask矩阵中加入值，便于进行位运算
Mask = Mask + 0
Mask.dtype = 'uint8'

# 将mask加入原图中，得到效果图
mask_image_1 = Mask[:, :, 0]
#mask_image_2 = Mask[:, :, 1]
#mask_image_3 = Mask[:, :, 2]

#mask_image = cv2.merge(mask_image_1, mask_image_2, mask_image_3)

#print(mask_image_1.dtype)

#ret, mask = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)

#Mask = np.load(filename2)
#Mask[:, :, 0]

mask_inv = cv2.bitwise_not(mask_image_1)

img2_fg = cv2.bitwise_and(image,image,mask = mask_inv)

cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

filename1 = 'PSPNet/color/591.png'
filename2 = 'Mask RCNN/1736.jpg [57 57 57 57 57 57 57 57] .npy'
filename = 'Orinigal_image/591.jpg'

color = [255, 120, 123, 0, 145, 123, 145, 125]

Mask_image = np.load(filename2)
Mask_image = Mask_image + 0
cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',PSPNet_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(PSPNet_image.shape)
print(Mask_image.shape)
#print(mask.shape)
Mask_image

"""
cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

mask = np.zeros(PSPNet_image.shape[:2], np.uint8)
mask[100:300, 100:400] = 255

mask.any()

PSPNet_image = cv2.imread(filename1, 0)
mask = Mask_image[:, :, 0] * 125
masked_image = cv2.bitwise_and(PSPNet_image, PSPNet_image, mask = mask)


plt.subplot(121), plt.imshow(PSPNet_image)
#plt.imshow(PSPNet_image)
plt.subplot(122), plt.imshow(masked_image)
plt.show()

filename1 = 'PSPNet/color/591.png'
PSPNet_image = cv2.imread(filename1, 1)

cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',PSPNet_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# 加载图像
img1 = cv2.imread('591.jpg')
img2 = cv2.imread('1736.jpg')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
# 取 roi 中与 mask 中不为零的值对应的像素的值,其他值为 0
# 注意这里必须有 mask=mask 或者 mask=mask_inv, 其中的 mask= 不能忽略
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

# 取 roi 中与 mask_inv 中不为零的值对应的像素的值,其他值为 0 。
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class_names[57]

"""
cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',image1)
#cv2.waitKey(0)
cv2.destroyAllWindows()
#image.shape[2] == 3
#image1.shape

#cv2.imshow('rose',image1)
"""

filename1 = 'PSPNet/color/591.png'
filename2 = 'Mask RCNN/1736.jpg [57 57 57 57 57 57 57 57] .npy'
filename = 'Orinigal_image/591.jpg'

color = [255, 120, 123, 0, 145, 123]

image1 = cv2.imread(filename1)
print(image1.shape)

image2 = np.load(filename2)

masked_image = apply_mask(image1, image2, 'gray')

plt.subplot(121), plt.imshow(image1, 'gray')
plt.subplot(122), plt.imshow(masked_image, 'gray')
plt.show()

print(image2.shape)

image2[:, :, 0]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

img = cv2.imread('591.jpg',2)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[200:400, 200:400] = 125
mask_inv = cv2.bitwise_not(mask)
masked_img = cv2.bitwise_and(img, img, mask = mask)
print(type(masked_img))

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask), plt.xlim([0,256])

plt.show()
