import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_manager import *

class Direct:

    def __init__(self, sess):

        self.batch_size = 1

        self.df = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 32, 2], name='distance_field')

        self.make_epn(self.df)

    def make_epn(self, data):

        nf = 40
        layers = 5

        # encoder network
        last = data
        for i in range(layers):
            enc = tf.layers.conv3d(inputs=last, filters=(2**i)*nf, kernel_size=[4, 4, 4],
                    padding="same", activation=tf.nn.relu)
            enc = tf.layers.max_pooling3d(inputs=enc, pool_size=[2, 2, 2], strides=2)
            enc = tf.layers.batch_normalization(enc)
            print 'Enc %d :' % (i), enc.get_shape().as_list()

            last = enc

        # dense layers
        enc_reshaped = tf.reshape(last, shape=[self.batch_size, -1])
        fc1 = tf.layers.dense(inputs=enc_reshaped, units=(2**(layers-1))*nf, activation=tf.nn.relu)
        print 'fc1 :', fc1.get_shape().as_list()
        fc2 = tf.layers.dense(inputs=fc1, units=(2**(layers-1))*nf, activation=tf.nn.relu)
        print 'fc2 :', fc2.get_shape().as_list()

        # decoder
        last = tf.reshape(fc2, shape=[self.batch_size, 1, 1, 1, (2**(layers-1))*nf])
        print 'reshaped :', last.get_shape().as_list()

        for i in range(layers):
            W = tf.get_variable('W_%d' % (i), [4, 4, 4, (2**(layers-i-2))*nf, (2**(layers-i-1))*nf])
            b = tf.get_variable('b_%d' % (i), [(2**(layers-i-2))*nf])
            dec = tf.nn.conv3d_transpose(last, W,
                    [self.batch_size, 2, 2, 2, (2**layers-i-1)*nf], strides=[1, 1, 1, 1, 1],
                    padding="SAME")
            dec = tf.nn.bias_add(dec, b)
            dec = tf.nn.relu(dec)
            print 'Dec %d: ' % (i), dec.get_shape().as_list()

            last = dec

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  direct = Direct(sess)
