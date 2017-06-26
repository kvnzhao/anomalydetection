#-*- coding:utf-8 -*-
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

FILE_FEAME = 200    #每个文件视频序列个数
PATECH_NUMBER = 176 #每帧图像被划分总块数


def readvideo(filename):
    """
    :param filename: 文件路径
    :return: 读取图片,单通道，灰度图像
    """
    img = cv2.imread(filename,0)
    if img is None:
        #print 'error open: ',filename, ' ......'
        pass
    return img


def getFlow(imPrev,imNew):
    flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    return flow

def calc_his(flow_map, t1, t2, p):
    """
    :param flow: 光流信息
    :param t1: 第一个阈值
    :param t2: 第二个阈值
    :p: 方向数
    :return: MHOF
    """
    #把光流转换成角度信息
    mag, ang = cv2.cartToPolar(flow_map[...,0], flow_map[...,1],angleInDegrees = 1)
    mag_flat = mag.flatten()
    ang_flat = ang.flatten()
    size = mag_flat.size
    for i in range(size):
        if(mag_flat[i] <= t1):
            mag_flat[i] = round(ang_flat[i]* p / 360) % p
        elif(mag_flat[i] > t1 and mag_flat[i] <= t2):
            mag_flat[i] = round(ang_flat[i] * p / 360) % p + p
        elif(mag_flat[i] > t2):
            mag_flat[i] = round(ang_flat[i] * p / 360) % p + 2 * p

    #计算MHOF直方图

    hist = np.zeros(3 * p, dtype='int16')
    b = set(mag_flat)
    for item in b:
        count = list(mag_flat).count(item)
        hist[int(item)] = count
        #print item, count

    #归一化
    #hist = hist *1.0 / size
    #print hist
    return hist

def extract_stacked_mhof(flow_patch, t1,t2,p):
    """
    提取每个时空块15*15*2的mhof特征
    :param flow_path: 多帧的光流图时空块
    :return: stacked mhof
    """

    temp_size = flow_patch.shape[0]
    stacked_hist = []
    for i in range(temp_size):
        hist = calc_his(flow_patch[i],t1,t2,p)
        stacked_hist.append(hist)
    mhof = np.array(stacked_hist).flatten()

    return mhof


def split_img(img, patch_size):
    """
    :param img: 输入数据
    :patch_size: 分割图像大小
    :return: 返回分割后的数组[n, x, y]
    """
    patch_row, patch_col = patch_size #子图的大小

    img_row = img.shape[0]
    img_col = img.shape[1]    #计算图像的行列数\
    img_chan = img.shape[2]

    row = img_row / patch_row
    col = img_col / patch_col   #计算子图的数目

    img_patch_collect = np.zeros((row * col, patch_row, patch_col,img_chan))
    n = 0
    for r in range(row):
        for c in range(col):
            img_patch_collect[n] = img[r*patch_row:(r+1)*patch_row,
                                   c*patch_col:(c+1)*patch_col, :].copy()
            n +=1

    return img_patch_collect


def resize_img(img):
    """
    
    :param img: 输入图像
    :return: 输出新图像（修改尺寸）
    """
    res = cv2.resize(img,(240,165),interpolation=cv2.INTER_CUBIC)
    return res

def save_flow_patch(file, file_number, save_filepath):
    """
    :param file: 文件路径
    :file_number: 文件数量
    :return: 保存光流到文件当中
    """
    global FILE_FEAME
    global PATECH_NUMBER
    i = 1   #文件目录
    j = 1   #图片目录
    preImg = None
    file_flow_patch = np.zeros((FILE_FEAME-1, PATECH_NUMBER, 15, 15, 2))
    while True:
        if i > file_number:
            print "all file have readed..."
            break
        filename = file + '%03d/%03d.tif'%(i,j) #生成图片路径
        img = readvideo(filename)

        if img is None:
            np.save(save_filepath + 'file_flow_patch_%03d.npy' % i, file_flow_patch)
            print "save file_flow_patch_%03d.npy successed ..."%i
            i +=1
            j = 1
            preImg = None
            print "begin next file %03d" % i
            continue

        img = resize_img(img)       #修改图像尺寸165*240
        #print img.shape[0]

        if j > 1:
            flow = getFlow(preImg, img)
            # cv2.imshow('imag',img)
            # cv2.imshow('flow',flow[:,:,0])
            # cv2.imshow('FLOW2',flow[:,:,1])
            # cv2.waitKey(500)
            flow_patch = split_img(flow, [15,15]) #得到每一帧光流子图
            file_flow_patch[j-2] = flow_patch

        preImg = img.copy()
        j +=1




def save_mhof(file, save_file, file_number, T1, T2, p, temp_size):
    """
    返回的是[fl,patch,f,m][文件，分块，帧，mhof特征]
    :param file: 存储光流的文件目录
    :return: mhof
    :temp_size: 每个时空块时间域长度
    """
    global FILE_FEAME
    global PATECH_NUMBER
    i = 1   #读取文件的目录
    spatial_temporal_mhof = np.zeros((file_number, PATECH_NUMBER,
                                      FILE_FEAME-temp_size, 3 * temp_size * p),dtype='int16')  # 存储每个文件的MHOF特征
    while True:
        if i > file_number:
            break
        filename = file + '_file_flow_patch_%03d.npy' % i
        file_flow_patch = np.load(filename)

        for j in range(PATECH_NUMBER):
            """
            将整个图像分割成176块，分别对每一块求mhof特征，存储
            2017.6.9修改，将图像块的阈值分为三部分，0-63,64-127,128-175
            
            """
            # (mhof延时间轴个数195，每个时空块mhof维度120)

            if j < 48:
                t1 = T1 - 0.4
            else:
                t1 = T1

            n_patch = j / 16
            t2 = T2 + n_patch * 0.2

            for k in range(FILE_FEAME - temp_size):
                #产生一个15*15*5的光流cubic
                flow_patch = file_flow_patch[k:k+temp_size,j]
                #计算15*15*5时空块的mhof特征，特征维度120维
                mhof = extract_stacked_mhof(flow_patch,t1, t2, p)
                spatial_temporal_mhof[i - 1, j, k] = mhof

        print file,' ',i, ' file processed...'
        i +=1

    np.save(save_file+'_mhof.npy',spatial_temporal_mhof)




if __name__ == "__main__":
    #print 'test split_img......'
    # a = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1,2,3,4,5,6,7,8,9,10],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1,2,3,4,5,6,7,8,9,10],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1,2,3,4,5,6,7,8,9,10],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[1,2,3,4,5,6,7,8,9,10]])
    #
    # img_split = split_img(a, [5,5])
    # print img_split.shape[2]
    train_file = "/home/kun/data/UCSD/UCSDped1/Train/Train"
    train_file_number = 34
    save_tain_filepath = "UCSD/UCSDped1/train/train_"
    test_file = "/home/kun/data/UCSD/UCSDped1/Test/Test"
    test_file_number = 36
    #save_test_filepath = "UCSD/UCSDped1/test/test_"
    #save_flow_patch(test_file,test_file_number,save_test_filepath)
    #c = np.load('UCSD/UCSDped1/train/train_file_flow_patch_%03d.npy'%1)
    #print c.shape
    #save_mhof(save_tain_filepath,train_file_number, 1, 3, 8, 5)
    c = np.load('UCSD/UCSDped1/train/train_mhof.npy')
    print c[0,0,0]





