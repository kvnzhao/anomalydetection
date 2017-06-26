#-*- coding:utf-8 -*-
import numpy as np

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