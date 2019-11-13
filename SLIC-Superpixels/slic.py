import cv2
import numpy as np

class SLIC:
    
    '''
    @step 块数量
    @nc 颜色距离参数
    '''
    def __init__(self, img, step, nc):
        self.img = img
        self.height, self.width = img.shape[:2]
        self._convert2LAB()
        self.step = step
        self.nc = nc
        self.ns = step
        self.FLT_MAX = 1000000
        self.ITERATIONS = 10
    
    def _convert2LAB(self):
        pass

img_rgb = cv2.imread("./Lenna.png")
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

cv2.imshow("1", img_rgb) 
cv2.imshow("2", img_lab) 
cv2.waitKey()
