import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

%matplotlib inline

image = cv2.imread('434162.jpg')
cv2.namedWindow('rose', cv2.WINDOW_NORMAL)
cv2.imshow('rose',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#image.shape[2] == 3
image.shape

# 遍历文件夹内文件名
image_name = os.listdir('Test_set')

for i in image_name:
    filename = 'Test_set/' + i
    #print(filename)
    image = cv2.imread(filename, 1)
    
    # 判断第三通道是否为3，为3则复制到对应目录
    if image.shape[2] == 3:
        shutil.copy(filename, 'rgb_image')

# 计算彩色图片张数
rgb_name = os.listdir('rgb_image')
len(rgb_name)

image2 = plt.imread('434162.jpg')
plt.imshow(image2)

img = cv2.imread('434162.jpg',0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
print(type(masked_img))

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()
