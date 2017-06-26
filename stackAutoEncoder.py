# -*- coding:utf-8 -*-

import time
from os.path import join as pjoin

import numpy as np
import tensorflow as tf

from flags import FLAGS
import inputdata

global patch


class AutoEncoder(object):
    """Generic deep autoencoder.
    自动编码器用于完整的训练周期，包括无监督的预训练层和最终的微调。 
    用户通过指定输入数量，每个层的隐藏单位数量和最终输出逻辑的数量来指定神经网络的结构。
    """
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"

    def __init__(self, shape, sess):
        """Autoencoder initializer
        修改网络形状，不加后面的logit layer，网络只有输入层，隐层1，...隐层n组成
        Args:
          shape: list of ints specifying
                  num input, hidden1 units,...hidden_n units
          sess: tensorflow session object to use
        """
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim]
        self.__num_hidden_layers = len(self.__shape) - 1

        self.__variables = {}
        self.__sess = sess

        self._setup_variables()

    @property
    def shape(self):
        return self.__shape

    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers

    @property
    def session(self):
        return self.__sess

    def __getitem__(self, item):
        """Get autoencoder tf variable
        得到autoencoder的tf的变量信息
    
        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.
    
        Args:
         item: string, variables internal name
        Returns:
         Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a tensorflow variable
        存储tf变量信息
    
        Args:
          key: string, name of variable
          value: tensorflow variable
        """
        self.__variables[key] = value

    def _setup_variables(self):
        with tf.name_scope("autoencoder_variables"):
            for i in xrange(self.__num_hidden_layers):  # 修改self._num_hidden_layers + 1为当前
                # Train weights
                name_w = self._weights_str.format(i + 1)
                w_shape = (self.__shape[i], self.__shape[i + 1])
                a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))  # 修改tf.mul
                w_init = tf.random_uniform(w_shape, -1 * a, a)
                self[name_w] = tf.Variable(w_init,
                                           name=name_w,
                                           trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
                b_init = tf.zeros(b_shape)
                self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

                if i < self.__num_hidden_layers:
                    # Hidden layer fixed weights (after pretraining before fine tuning)
                    # 每一层训练好的weigts权重，在fine-train之前

                    self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                          name=name_w + "_fixed",
                                                          trainable=False)

                    # Hidden layer fixed biases
                    self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                          name=name_b + "_fixed",
                                                          trainable=False)

                    # Pretraining output training biases
                    name_b_out = self._biases_str.format(i + 1) + "_out"
                    b_shape = (self.__shape[i],)
                    b_init = tf.zeros(b_shape)
                    self[name_b_out] = tf.Variable(b_init,
                                                   trainable=True,
                                                   name=name_b_out)

    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]

    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]

    def get_variables_to_init(self, n):
        """Return variables that need initialization
            返回需要初始化的变量
    
        This method aides in the initialization of variables
        before training begins at step n. The returned
        list should be than used as the input to
        tf.initialize_variables
    
        Args:
          n: int giving step of training
          训练的第n步
        """
        assert n > 0
        assert n <= self.__num_hidden_layers + 1  # 修改self._num_hidden_layers + 1

        vars_to_init = []
        if n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n))
            vars_to_init.append(self._b(n))
            vars_to_init.append(self._b(n, "_out"))

        if 1 < n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n - 1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))

        if n == self.__num_hidden_layers + 1:
            vars_to_init = [self._w(n - 1, "_fixed"), self._b(n - 1, "_fixed")]

        return vars_to_init

    def save_variable(self, n):

        if n == 1:
            print self._w(n)

        if n == 2:
            print self._w(n - 1, '_fixed')

    @staticmethod
    def _activate(x, w, b, transpose_w=False):
        y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        return y

    def pretrain_net(self, input_pl, n, is_target=False):
        """Return net for step n training or target net
        预训练的网络，返回预训练第n层网络的输出值或者是预训练的目标值
    
        Args:
          input_pl:  tensorflow placeholder of AE inputs
          n:         int specifying pretrain step
          is_target: bool specifying if required tensor
                      should be the target tensor
        Returns:
          Tensor giving pretraining net or pretraining target
        """
        assert n > 0
        assert n <= self.__num_hidden_layers

        last_output = input_pl
        for i in xrange(n - 1):
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")

            last_output = self._activate(last_output, w, b)

        if is_target:
            # last_output = tf.maximum(last_output, 1.e-9)
            # last_output = tf.minimum(last_output, 1 - 1.e-9)
            return last_output

        last_output = self._activate(last_output, self._w(n), self._b(n))

        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                             transpose_w=True)
        out = tf.maximum(out, 1.e-9)
        out = tf.minimum(out, 1 - 1.e-9)
        return out

    def finetuning_net(self, input_pl, is_output = False):
        """Get the supervised fine tuning net
        这部分函数需要修改，函数本身为有标签数据的分类，需要改为无监督的fine-train
        ......!!!!
        Args:
          input_pl: tf placeholder for ae input data
        Returns:
          Tensor giving full ae net
    
        """

        last_output = input_pl
        for i in xrange(self.__num_hidden_layers):
            w = self._w(i + 1)
            b = self._b(i + 1)

            last_output = self._activate(last_output, w, b)

        if is_output:
            return last_output

        for i in xrange(self.__num_hidden_layers, 0, -1):
            w = self._w(i)
            b = self._b(i, "_out")

            last_output = self._activate(last_output, w, b, transpose_w=True)

        last_output = tf.maximum(last_output, 1.e-9)
        last_output = tf.minimum(last_output, 1 - 1.e-9)

        return last_output


def training(loss, learning_rate, loss_key=None):
    """Sets up the training Ops.
    设置训练代价loss
    Creates a summarizer to track the loss over time in TensorBoard.
    创建一个sumarizer来跟踪误差在TensorBoard上
  
    Creates an optimizer and applies the gradients to all trainable variables.
  
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
  
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
      loss_key: int giving stage of pretraining so we can store
                  loss summaries for each pretraining stage
  
    Returns:
      train_op: The Op for training.
    """
    # if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    # loss_summaries[loss_key] = tf.scalar_summary(loss.op.name, loss)
    # else:
    # tf.scalar_summary(loss.op.name, loss)
    # for var in tf.trainable_variables():
    # tf.histogram_summary(var.op.name, var)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step


def loss_x_entropy(output, target):
    """Cross entropy loss
    损失计算
    See https://en.wikipedia.org/wiki/Cross_entropy
  
    Args:
      output: tensor of net output
      target: tensor of net we are trying to reconstruct
    Returns:
      Scalar tensor of cross entropy
    """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        # 修改tf.mul = tf.multiply
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                           target_tf),
                               tf.multiply(tf.log(1 - net_output_tf),
                                           (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                   name='xentropy_mean')


def main_unsupervised():
    global patch
    with tf.Graph().as_default() as g:
        sess = tf.Session()

        num_hidden = FLAGS.num_hidden_layers
        ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                            for j in xrange(num_hidden)]
        ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes  # 去掉[FLAGS.num_classes]

        ae = AutoEncoder(ae_shape, sess)

        data = inputdata.read_data_sets(FLAGS.train_data_dir, FLAGS.test_data_dir, patch)
        print 'unsupervised',patch
        num_train = data.train.num_examples

        learning_rates = {j: getattr(FLAGS,
                                     "pre_layer{0}_learning_rate".format(j + 1))
                          for j in xrange(num_hidden)}

        noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
                 for j in xrange(num_hidden)}

        for i in xrange(len(ae_shape) - 1):  # 修改-2 为 -1
            n = i + 1
            with tf.variable_scope("pretrain_{0}".format(n)):
                input_ = tf.placeholder(dtype=tf.float32,
                                        shape=(FLAGS.batch_size, ae_shape[0]),
                                        name='ae_input_pl')
                target_ = tf.placeholder(dtype=tf.float32,
                                         shape=(FLAGS.batch_size, ae_shape[0]),
                                         name='ae_target_pl')
                layer = ae.pretrain_net(input_, n)

                with tf.name_scope("target"):
                    target_for_loss = ae.pretrain_net(target_, n, is_target=True)

                loss = loss_x_entropy(layer, target_for_loss)
                train_op, global_step = training(loss, learning_rates[i], i)

                vars_to_init = ae.get_variables_to_init(n)
                vars_to_init.append(global_step)  # global_step 没有看懂
                sess.run(tf.variables_initializer(vars_to_init))

                print("\n\n")
                print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
                print("|---------------|---------------|---------|----------|")

                for step in xrange(FLAGS.pretraining_epochs * num_train):
                    feed_dict = inputdata.fill_feed_dict_ae(data.train, input_, target_, noise[i])
                    # batch所训练的分块

                    loss_summary, loss_value = sess.run([train_op, loss],
                                                        feed_dict=feed_dict)

                    if step % 100 == 0:
                        output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |" \
                            .format(step, loss_value, n, step / num_train + 1)  # 修改// 为单个/

                        print(output)

                    # if n == 1:
                    # print(sess.run(ae._w(n)))
                    # if n == 2:
                    # print(sess.run(ae._w(n-1,'_fixed')))

    return ae


def main_finetuning(ae):
    global patch
    with ae.session.graph.as_default():
        sess = ae.session
        input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                     FLAGS.image_pixels),
                                  name='input_pl')
        input_o = tf.placeholder(tf.float32, shape=(None,FLAGS.image_pixels))

        # target_o = tf.placeholder(tf.float32,shape=(1,FLAGS.image_pixels),
        #                            name='target_pl')

        output_pl = ae.finetuning_net(input_pl)  # 修改supervised_net
        target_o = ae.finetuning_net(input_o, is_output=True)

        data = inputdata.read_data_sets(FLAGS.train_data_dir, FLAGS.test_data_dir, patch)
        print 'fintuning',patch
        num_train = data.train.num_examples

        # finetuning也是无监督的，所以label信息可能是用不到


        # loss = loss_supervised(logits, labels_placeholder)
        loss = loss_x_entropy(output_pl, input_pl)  # 修改finetuning的loss函数

        train_op, global_step = training(loss, FLAGS.supervised_learning_rate)  # ！！！！这里的learning_rete 需要修改
        vars_to_init = ae.get_variables_to_init(ae.num_hidden_layers + 1)
        vars_to_init.append(global_step)

        sess.run(tf.variables_initializer(vars_to_init))
        steps = FLAGS.finetuning_epochs * num_train
        for step in xrange(steps):
            start_time = time.time()

            feed_dict = inputdata.fill_feed_dict(data.train, input_pl)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.6f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.

        num_train = data.train.num_examples
        output_encoder = np.zeros((num_train, ae.shape[ae.num_hidden_layers]),dtype='float32')
        for i in range(num_train):
            encoder_mhof = sess.run(target_o,feed_dict={input_o:data.test.data_mhof[i:i+1]})
            output_encoder[i] = encoder_mhof

        np.save('UCSD/UCSDped1/train/train_encoder_feature/train_encoder_patch_%d.npy' % patch, output_encoder)

        num_test = data.test.num_examples
        output_encoder = np.zeros((num_test,ae.shape[ae.num_hidden_layers]), dtype='float32')
        for i in range(num_test):
            encoder_mhof = sess.run(target_o, feed_dict={input_o: data.test.data_mhof[i:i + 1]})
            output_encoder[i] = encoder_mhof

        np.save('UCSD/UCSDped1/test/test_encoder_feature/test_encoder_patch_%d.npy'%patch, output_encoder)
        sess.close()


if __name__ == '__main__':
    patch = 90
    print 'begin to patch ',patch,'......'
    for i in range(2):
        ae = main_unsupervised()
        main_finetuning(ae)
        patch += 1
