#-*- coding:utf-8 -*-
"""
将已有的特征生成tensorflow可以直接调用的格式
"""

import numpy as np
from flags import FLAGS

def extract_mhof(filename):
    """ 提取从文件中提取HOF特征 [num,x] """
    #filename文件路径，num文件个数
    print('extract mhof......')
    print filename
    data_mhof = np.load(filename)

    data_mhof = np.multiply(data_mhof, 1.0/225.0)
    return data_mhof

class DataSet(object):
    def __init__(self, data_mhof):
        self._num_examples = data_mhof.shape[0]
        self._data_mhof = data_mhof
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data_mhof(self):
        return self._data_mhof

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle = True):
        """返回 next bach_size 样本,patch 表示图像块位于图像中的位置"""
        start = self._index_in_epoch
        #shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data_mhof = self.data_mhof[perm0]

        #go to the next epoch
        if start + batch_size > self._num_examples:
            #Finished epoch
            self._epochs_completed += 1
            #Get the rest examples in this epoch
            rest_num_examples = self._num_examples -start
            hof_rest_part = self._data_mhof[start:self._num_examples]

            #shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data_mhof = self.data_mhof[perm]

            start = 0
            self._index_in_epoch = batch_size -rest_num_examples
            end = self._index_in_epoch

            hof_new_part = self._data_mhof[start:end]

            return np.concatenate((hof_rest_part, hof_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            return self._data_mhof[start:end]


def read_region_data_sets(train_dir, test_dir, patch):
    class DataSets(object):
        pass

    data_sets = DataSets()

    #读取训练和测试数据集
    n_train = []
    n_test = []

    for i in range(9):
        if i == 0:
            patch_n = patch -17
            if patch_n < 0 or (patch / 16) != (patch_n /16 +1) :
                patch_n = -1

        if i == 1:
            patch_n = patch - 16
            if patch_n < 0:
                patch_n = -1

        if i == 2:
            patch_n = patch - 15
            if patch_n < 0 or (patch / 16) != (patch_n / 16 + 1):
                patch_n = -1

        if i == 3:
            patch_n = patch - 1
            if (patch / 16) != (patch_n / 16):
                patch_n = -1

        if i == 4:
            patch_n = patch

        if i == 5:
            patch_n = patch + 1
            if (patch / 16) != (patch_n / 16):
                patch_n = -1

        if i == 6:
            patch_n = patch + 15
            if patch_n > 175 or (patch / 16) != (patch_n / 16 - 1):
                patch_n = -1

        if i == 7:
            patch_n = patch + 16
            if patch_n > 175:
                patch_n = -1

        if i == 8:
            patch_n = patch + 17
            if patch_n > 175 or (patch_n / 16 -1) != (patch / 16):
                patch_n = -1


        if patch_n != -1:
            train_data = extract_mhof(train_dir + '/train_patch_%d.npy' % patch_n)
            test_data = extract_mhof(test_dir + '/test_patch_%d.npy' % patch_n)

            n_train.append(train_data)
            n_test.append(test_data)

    train_mhof = np.vstack(n_train)
    test_mhof = np.vstack(n_test)


    data_sets.train = DataSet(train_mhof)
    data_sets.test = DataSet(test_mhof)


    return data_sets

def read_data_sets(train_dir, test_dir, patch):
    class DataSets(object):
        pass

    data_sets = DataSets()

    #读取训练和测试数据集
    train_mhof = extract_mhof(train_dir + '/train_patch_%d.npy'%patch)
    test_mhof = extract_mhof(test_dir + '/test_patch_%d.npy'%patch)

    data_sets.train = DataSet(train_mhof)
    data_sets.test = DataSet(test_mhof)

    return data_sets

#看一下如何添加噪声
def _add_noise(x, rate):
  x_cp = np.copy(x)
  pix_to_drop = np.random.rand(x_cp.shape[0],
                                  x_cp.shape[1]) < rate
  x_cp[pix_to_drop] = FLAGS.zero_bound
  return x_cp

def fill_feed_dict_ae(data_set, input_pl, target_pl, noise=None):
    input_feed = data_set.next_batch(FLAGS.batch_size)
    target_feed = input_feed.copy()
    # batch_size 每一個batch的大小，patch 訓練當前哪個分块
    if noise:
      input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed
    }
    return feed_dict

def fill_feed_dict(data_set, input_pl, noise=None):
    input_feed = data_set.next_batch(FLAGS.batch_size)
    # batch_size 每一個batch的大小，patch 訓練當前哪個分块
    if noise:
      input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
    }
    return feed_dict