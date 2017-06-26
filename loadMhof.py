#-*- coding:utf-8 -*-

"""
从文件中加载mhof特征
"""

import numpy as np

import matplotlib.pyplot as plt

import inputdata
import inputdatas
import flags as FLAGS

train_data_dir = '/home/kun/data/UCSD/UCSDped1/Train/train4_feature'
test_data_dir = '/home/kun/data/UCSD/UCSDped1/Test/test4_feature'

mhof_tain_filepath = "/home/kun/data/UCSD/UCSDped1/Train/train4_mhof.npy"
mhof_test_filepath = "/home/kun/data/UCSD/UCSDped1/Test/test4_mhof.npy"

ta = np.load(mhof_tain_filepath)
te = np.load(mhof_test_filepath)

print ta.shape

mhof_re = np.zeros((ta.shape[0] * ta.shape[2], ta.shape[1], ta.shape[3]), dtype = 'int16')

k = 0

for i in range(ta.shape[0]):
    for j in range(ta.shape[2]):
        mhof_re[k] = ta[i,:,j]
        k +=1

for i in range(ta.shape[1]):
    np.save("/home/kun/data/UCSD/UCSDped1/Train/train4_feature/train_patch_%d.npy"%i, mhof_re[:,i])


mhof_re = np.zeros((te.shape[0] * te.shape[2], te.shape[1], te.shape[3]), dtype = 'int16')

k = 0

for i in range(te.shape[0]):
    for j in range(te.shape[2]):
        mhof_re[k] = te[i,:,j]
        k +=1

for i in range(te.shape[1]):
    np.save("/home/kun/data/UCSD/UCSDped1/Test/test4_feature/test_patch_%d.npy"%i, mhof_re[:,i])

# patch1_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_feature/train_patch_175.npy"
#
# patch1 = np.load(patch1_path)
#
# print patch1.shape
#
# e = np.load("/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_63.npy")
# a = np.load("/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_1.npy")

# test_path = "/home/kun/data/UCSD/UCSDped1/Test/test1_encoder_feature/test_encoder_patch_%d.npy" % 5
#
# #test_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_%d.npy" % 5
#
# test_x = np.load(test_path)
#
# print test_x.shape

# ucsd = inputdatas.read_data_sets(train_data_dir, test_data_dir, 1)
#
# print ucsd.test.num_examples

# c = np.load("/home/kun/data/UCSD/UCSDped1/Test/test_encoder_feature/test_encoder_patch_91.npy")


