import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

def show_image(image):
    #for i in range(N):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('Image_Fusion/2054.png')

show_image(image)

#img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(image,240,250)
show_image(edges)


plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
