import cv2
import numpy as np
from skimage.measure import compare_ssim

def SSIM(img1,img2):    
    return compare_ssim(img1,img2)

if __name__ == '__main__':
    img1=np.random.randn(32,32,32)
    img2=np.random.randn(32,32,32)
    print(SSIM(img1,img2))