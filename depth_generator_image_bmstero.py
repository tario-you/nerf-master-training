# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read two input images
imgL = cv2.imread('aloeL.jpg', 0)
imgR = cv2.imread('aloeR.jpg', 0)

# Initiate and StereoBM object
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

# compute the disparity map
disparity = stereo.compute(imgL, imgR)
disparity1 = stereo.compute(imgR, imgL)
plt.imshow(disparity, 'gray')
plt.show()
help(plt.imsave)
