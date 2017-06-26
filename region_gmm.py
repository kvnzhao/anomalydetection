#-*- coding:utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import cv2
import calHistogramOpticalFlow as chof
#读取数据

save_test = 2

y_pred = np.zeros((176,7020))
red = np.full((15, 15, 3), [0, 0, 255], dtype='uint8')


for patch in range(1, 4):

    if patch == 1:
        n_train = []
        for i in range(64):
            patch_n = i
            train_path = "/home/kun/data/UCSD/UCSDped1/Train/train2_encoder_feature/train_encoder_patch_%d.npy" % patch_n

            train_data = np.load(train_path)
            n_train.append(train_data)

        train_x = np.vstack(n_train)

    if patch == 2:
        n_train = []
        for i in range(64,128):
            patch_n = i
            train_path = "/home/kun/data/UCSD/UCSDped1/Train/train2_encoder_feature/train_encoder_patch_%d.npy" % patch_n

            train_data = np.load(train_path)
            n_train.append(train_data)

        train_x = np.vstack(n_train)

    if patch == 3:
        n_train = []
        for i in range(128,176):
            patch_n = i
            train_path = "/home/kun/data/UCSD/UCSDped1/Train/train2_encoder_feature/train_encoder_patch_%d.npy" % patch_n

            train_data = np.load(train_path)
            n_train.append(train_data)

        train_x = np.vstack(n_train)

    #训练过程

    print "begin train gmm %d"%patch

    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
    gmm.fit(train_x)

    #测试过程

    if patch == 1:
        for i in range(64):
            patch_n = i
            test_path = "/home/kun/data/UCSD/UCSDped1/Test/test2_encoder_feature/test_encoder_patch_%d.npy" % patch_n
            test_x = np.load(test_path)
            y_pred_test = gmm.score_samples(test_x)
            y_pred[patch_n] = y_pred_test
            print "finish svm %d" % patch_n

    if patch == 2:
        for i in range(64,128):
            patch_n = i
            test_path = "/home/kun/data/UCSD/UCSDped1/Test/test2_encoder_feature/test_encoder_patch_%d.npy" % patch_n
            test_x = np.load(test_path)
            y_pred_test = gmm.score_samples(test_x)
            y_pred[patch_n] = y_pred_test
            print "finish svm %d" % patch_n

    if patch == 3:
        for i in range(128,176):
            patch_n = i
            test_path = "/home/kun/data/UCSD/UCSDped1/Test/test2_encoder_feature/test_encoder_patch_%d.npy" % patch_n
            test_x = np.load(test_path)
            y_pred_test = gmm.score_samples(test_x)
            y_pred[patch_n] = y_pred_test
            print "finish svm %d" % patch_n


fps = 24

videoWrite = cv2.VideoWriter('/home/kun/data/UCSD/UCSDped1/Test/test%d.avi'%save_test, cv2.cv.CV_FOURCC(*'MJPG'), fps, (240, 165))

file_frame = np.zeros((36,195),dtype='uint8')

for i in range(y_pred.shape[1]):
    error_flag = 0
    file = i / 195 + 1
    frame = i % 195 + 1
    image_path = "/home/kun/data/UCSD/UCSDped1/Test/Test%03d/%03d.tif"%(file, frame)
    img = cv2.imread(image_path, 1)

    img = chof.resize_img(img)

    for j in range(176):
        n = j #第j个patch

        r = n / 16
        c = n % 16

        if y_pred[n][i] < 0:
            error_flag = error_flag + 1
            cv2.addWeighted(img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15], 1, red, 1, 0.0,
                            img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15])
    if error_flag >= 1:
        file_frame[file - 1, frame - 1] = 1
    # cv2.imshow('img', img)
    # cv2.waitKey(30)
    videoWrite.write(img)

np.save("/home/kun/data/UCSD/UCSDped1/Test/test_result/%d.npy"%save_test,file_frame)
videoWrite.release()

