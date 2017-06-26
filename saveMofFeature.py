#-*- coding:utf-8 -*-

"""
保存训练和测试的mhof特征[filenumber,patch,frame_n,feature]
"""

import numpy as np
import cv2
import calHistogramOpticalFlow as chof

#train_file = "/home/kun/data/UCSD/UCSDped1/Train/Train"
train_file_number = 34
train_flow_filepath = "UCSD/UCSDped1/train/train"
train_mhof_filepath = "/home/kun/data/UCSD/UCSDped1/Train/train4"
test_file = "/home/kun/data/UCSD/UCSDped1/Test/Test"
test_file_number = 36
test_flow_filepath = "UCSD/UCSDped1/test/test"
test_mhof_filepath = "/home/kun/data/UCSD/UCSDped1/Test/test4"

chof.save_mhof(train_flow_filepath,train_mhof_filepath, train_file_number, 1, 1.2, 8, 5)
chof.save_mhof(test_flow_filepath,test_mhof_filepath, test_file_number, 1, 1.2, 8, 5)