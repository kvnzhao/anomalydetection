#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import inputdata
#import flags as FLAGS

train_data_dir = '/home/kun/data/UCSD/UCSDped1/Train/train4_feature'
test_data_dir = '/home/kun/data/UCSD/UCSDped1/Test/test4_feature'


learning_rate = 0.01
training_epoch = 500
batch_size = 100

n_hidden = 30
n_input = 120

X = tf.placeholder(tf.float32, [None,n_input])

W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

cost = tf.reduce_mean(tf.pow(X - decoder, 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.9).minimize(cost)

#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

for i in range(176):
    patch = i
    ucsd = inputdata.read_region_data_sets(train_data_dir, test_data_dir, patch)
    ucsd_0 = inputdata.read_data_sets(train_data_dir, test_data_dir, patch)

    print ('begin patch = ', patch)
    with tf.Session() as sess:

        sess.run(init)

        total_batch = int(ucsd.train.num_examples/batch_size)

        for epoch in range(training_epoch):
            total_cost = 0

            for i in range(total_batch):
                batch_xs = ucsd.train.next_batch(batch_size)
                _, cost_val = sess.run([optimizer, cost],
                                       feed_dict={X:batch_xs})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                  'average.cost = ', '{:.6f}'.format(total_cost / total_batch))

        #output

        num_train = ucsd_0.train.num_examples
        output_train_encoder = np.zeros((num_train, n_hidden), dtype='float32')

        for i in range(num_train):
            encoder_mhof = sess.run(encoder, feed_dict={X: ucsd_0.train.data_mhof[i:i + 1]})
            output_train_encoder[i] = encoder_mhof

        np.save('/home/kun/data/UCSD/UCSDped1/Train/train400_encoder_feature/train_encoder_patch_%d.npy' % patch, output_train_encoder)

        num_test = ucsd_0.test.num_examples
        output_test_encoder = np.zeros((num_test, n_hidden), dtype='float32')
        for i in range(num_test):
            encoder_mhof = sess.run(encoder, feed_dict={X: ucsd_0.test.data_mhof[i:i + 1]})
            output_test_encoder[i] = encoder_mhof

        np.save('/home/kun/data/UCSD/UCSDped1/Test/test400_encoder_feature/test_encoder_patch_%d.npy' % patch, output_test_encoder)


