#-*- coding:utf-8 -*-
import numpy as np
import calHistogramOpticalFlow as chof
import cv2
import matplotlib.pyplot as plt

c = np.load('UCSD/UCSDped1/test/test_file_flow_patch_001.npy')

print c.shape

n = c[:,137]

mag, ang = cv2.cartToPolar(n[...,0], n[...,1],angleInDegrees = 1)

mag = mag.flatten()

plt.hist(mag,50, normed=1, histtype='bar',facecolor = 'blue', alpha = 0.5)
plt.show()


print mag.shape


