#-*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import cv2
import calHistogramOpticalFlow as chof

train_path = "/home/kun/data/UCSD/UCSDped1/Train/train_encoder_feature/train_encoder_patch_104.npy"
test_path = "/home/kun/data/UCSD/UCSDped1/Test/test_encoder_feature/test_encoder_patch_104.npy"

# train_path = "UCSD/UCSDped1/train/train_feature/train_patch_90.npy"
# test_path = "UCSD/UCSDped1/test/test_feature/test_patch_90.npy"

train_x = np.load(train_path)
test_x = np.load(test_path)

clf = svm.OneClassSVM(nu = 0.01, kernel='rbf', gamma=0.6)
clf.fit(train_x)

y_pred_train = clf.predict(train_x)
y_pred_test = clf.predict(test_x)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

print n_error_train

# for i in range(y_pred_test.shape[0]):
#     if y_pred_test[i] == -1:
#         file = i / 195 + 1
#         frame = i % 195 + 1
#         image_path = "/home/kun/data/UCSD/UCSDped1/Test/Test%03d/%03d.tif"%(file, frame)
#         img = cv2.imread(image_path, 1)
#
#         img = chof.resize_img(img)
#
#         red = np.full((15, 15, 3), [0, 0, 255], dtype='uint8')
#
#         n = 104
#
#         r = n / 16
#         c = n % 16
#
#         cv2.addWeighted(img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15], 1, red, 1, 0.0,
#                     img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15])
#
#         cv2.imshow('img', img)
#         cv2.waitKey(300)

