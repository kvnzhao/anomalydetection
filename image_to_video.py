#-*- coding:utf-8 -*-
import os
import cv2

img_root = "/home/kun/data/UCSD/UCSDped1/Test/"
fps = 24

videoWrite = cv2.VideoWriter('/home/kun/data/UCSD/UCSDped1/Test/testvideo003.avi', cv2.cv.CV_FOURCC(*'MJPG'), fps,(238,158))

for i in range(1,200):
    image_path = "/home/kun/data/UCSD/UCSDped1/Test/Test003/%03d.tif"%i
    frame = cv2.imread(image_path)

    # cv2.imshow('r',frame)
    # cv2.waitKey(100)
    videoWrite.write(frame)

videoWrite.release()