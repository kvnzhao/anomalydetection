#-*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import cv2
import calHistogramOpticalFlow as chof

y_pred = np.zeros((176,7020))
red = np.full((15, 15, 3), [0, 0, 255], dtype='uint8')

for patch in range(176):

    # if patch == 1:
    #     n_train = []
    #     for i in range(64):
    #         patch_n = i
    #         train_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_%d.npy" % patch_n
    #
    #         train_data = np.load(train_path)
    #         n_train.append(train_data)
    #
    #     train_x = np.vstack(n_train)
    #
    # if patch == 2:
    #     n_train = []
    #     for i in range(64,128):
    #         patch_n = i
    #         train_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_%d.npy" % patch_n
    #
    #         train_data = np.load(train_path)
    #         n_train.append(train_data)
    #
    #     train_x = np.vstack(n_train)
    #
    # if patch == 3:
    #     n_train = []
    #     for i in range(128,176):
    #         patch_n = i
    #         train_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_encoder_feature/train_encoder_patch_%d.npy" % patch_n
    #
    #         train_data = np.load(train_path)
    #         n_train.append(train_data)
    #
    #     train_x = np.vstack(n_train)


    #test_path = "/home/kun/data/UCSD/UCSDped1/Test/test_encoder_feature/test_encoder_patch_%d.npy"%patch

    # train_path = "UCSD/UCSDped1/train/train_feature/train_patch_90.npy"
    # test_path = "UCSD/UCSDped1/test/test_feature/test_patch_90.npy"
    #
    # train_x = np.load(train_path)
    # test_x = np.load(test_path)


    train_path = "/home/kun/data/UCSD/UCSDped1/Train/train_encoder_feature/train_encoder_patch_%d.npy"%patch
    test_path = "/home/kun/data/UCSD/UCSDped1/Test/test_encoder_feature/test_encoder_patch_%d.npy"%patch

    # train_path = "/home/kun/data/UCSD/UCSDped1/Train/train1_feature/train_patch_%d.npy"%patch
    # test_path = "/home/kun/data/UCSD/UCSDped1/Test/test1_feature/test_patch_%d.npy"%patch

    train_x = np.load(train_path)
    test_x = np.load(test_path)


    print "begin train region: %d"%patch
    clf = svm.OneClassSVM(nu = 0.01, kernel='rbf', gamma=0.6)
    clf.fit(train_x)

    y_pred_train = clf.predict(train_x)

    # if patch == 1:
    #     for i in range(64):
    #         patch_n = i
    #         test_path = "/home/kun/data/UCSD/UCSDped1/Test/test1_encoder_feature/test_encoder_patch_%d.npy" % patch_n
    #         test_x = np.load(test_path)
    #         y_pred_test = clf.predict(test_x)
    #         y_pred[patch_n] = y_pred_test
    #         print "finish svm %d" % patch_n
    #
    # if patch == 2:
    #     for i in range(64,128):
    #         patch_n = i
    #         test_path = "/home/kun/data/UCSD/UCSDped1/Test/test1_encoder_feature/test_encoder_patch_%d.npy" % patch_n
    #         test_x = np.load(test_path)
    #         y_pred_test = clf.predict(test_x)
    #         y_pred[patch_n] = y_pred_test
    #         print "finish svm %d" % patch_n
    #
    # if patch == 3:
    #     for i in range(128,176):
    #         patch_n = i
    #         test_path = "/home/kun/data/UCSD/UCSDped1/Test/test1_encoder_feature/test_encoder_patch_%d.npy" % patch_n
    #         test_x = np.load(test_path)
    #         y_pred_test = clf.predict(test_x)
    #         y_pred[patch_n] = y_pred_test
    #         print "finish svm %d" % patch_n


    y_pred_test = clf.predict(test_x)
    #
    y_pred[patch] = y_pred_test

    #print "finish svm %d"%patch_n

    # n_error_train = y_pred_train[y_pred_train == -1].size
    # n_error_test = y_pred_test[y_pred_test == -1].size

#np.save("/home/kun/data/UCSD/UCSDped1/Test/test_single_oneclassSVM.npy", y_pred)

for i in range(y_pred.shape[1]):
    file = i / 195 + 1
    frame = i % 195 + 1
    image_path = "/home/kun/data/UCSD/UCSDped1/Test/Test%03d/%03d.tif"%(file, frame)
    img = cv2.imread(image_path, 1)

    img = chof.resize_img(img)

    for j in range(176):
        n = j #第j个patch

        r = n / 16
        c = n % 16

        if y_pred[n][i] == -1:
            cv2.addWeighted(img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15], 1, red, 1, 0.0,
                            img[r * 15:(r + 1) * 15, c * 15:(c + 1) * 15])

    cv2.imshow('img', img)
    cv2.waitKey(100)

