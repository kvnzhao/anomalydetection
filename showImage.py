#-*- coding:utf-8 -*-
import numpy as np
import cv2
import calHistogramOpticalFlow as chof
for patch in range(100, 101):
    print "patch %d"%patch
    filename = '/home/kun/data/UCSD/UCSDped1/Test/Test002/%03d.tif'%patch
    img = cv2.imread(filename,1)

    img = chof.resize_img(img)

    red = np.full((15,15,3),[0,0,255],dtype='uint8')

    n = 63



    r = n/16
    c = n%16

    cv2.addWeighted(img[r*15:(r+1)*15, c*15:(c+1)*15], 1, red, 1, 0.0,
                                    img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15])

    cv2.imshow('img',img)
    cv2.waitKey(70000)
    # print img.shape

