import numpy as np
import math

def PSNR(img1,img2):
    #compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1/1.0-img2/1.0)**2)
    #compute psnr
    if mse < 1e-10:
        return 100
    psnr = 10*math.log10(1.0/mse)
    return psnr