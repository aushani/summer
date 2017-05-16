import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_manager import *

class ALI:

  def __init__(self, sess, data_dim, z_dim):

    self.dm = DataManager('/ssd/nclt_og_data/2012-08-04/', data_dim)
    #self.dm = DataManager('/home/aushani/ramdisk/', data_dim)
    self.batch_size = 8

    self.sess = sess

    self.data_dim = data_dim
    self.z_dim = z_dim

    self.real_data = tf.placeholder(tf.float32, [None] + self.data_dim, name='real_data')
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

    # Generator
    self.encoder = self.make_encoder(self.real_data)
    self.decoder = self.make_decoder(self.z)

    # Discriminator
    self.x_discriminator_real = self.make_x_discriminator(self.real_data)
    self.x_discriminator_syth = self.make_x_discriminator(self.decoder, reuse=True)

    self.z_discriminator_real = self.make_z_discriminator(self.encoder)
    self.z_discriminator_syth = self.make_z_discriminator(self.z, reuse=True)

    self.discriminator_real = self.x_discriminator_real + self.z_discriminator_real
    self.discriminator_syth = self.x_discriminator_syth + self.z_discriminator_syth

    # Loss Functions
    self.encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.discriminator_real, labels=tf.ones_like(self.discriminator_real)))
    self.decoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.discriminator_syth, labels=tf.zeros_like(self.discriminator_syth)))

    self.generator_loss = self.encoder_loss + self.decoder_loss
    self.generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)

    self.discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=self.discriminator_real, labels=tf.zeros_like(self.discriminator_real)))
    self.discriminator_loss_syth = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=self.discriminator_syth, labels=tf.ones_like(self.discriminator_syth)))

    self.discriminator_loss = self.discriminator_loss_real + self.discriminator_loss_syth
    self.discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)


    # Evalualation
    self.query_data = tf.placeholder(tf.float32, [None] +  self.data_dim, name='query_data')
    self.query_z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'query_z')
    self.query_logits = self.make_x_discriminator(self.query_data, reuse=True) + self.make_z_discriminator(self.query_z, reuse=True)
    self.query_softmax = tf.nn.softmax(self.query_logits)

  def make_encoder(self, x):

    stage_layers = [16, 32, 64, 128, 256]

    prev_res = x

    with tf.variable_scope('Encoder'):
      for i in range(len(stage_layers)):
        H_conv = tf.layers.conv3d(inputs=prev_res, filters=stage_layers[i], kernel_size=[4, 4, 4],
                                    padding="same", activation=tf.nn.relu)

        H_i = tf.layers.max_pooling3d(inputs=H_conv, pool_size=[2, 2, 2], strides=2)

        print 'Encoder conv layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Now add dense layer
      last_shape = prev_res.get_shape().as_list()
      n = last_shape[1]*last_shape[2]*last_shape[3]*last_shape[4]
      flat = tf.reshape(prev_res, shape=[-1, n])
      dense = tf.layers.dense(inputs=flat, units=self.z_dim)

      print 'Encoder dense layer:', dense.get_shape().as_list()

      return dense


  def make_decoder(self, z):
    stage_layers = [256, 128, 64, 32, 16]
    base_dim = self.data_dim[0]/(2**len(stage_layers))

    with tf.variable_scope('Decoder'):
      # Dense layer
      dense_flat = tf.layers.dense(inputs=z, units=stage_layers[0]*base_dim*base_dim*base_dim)
      dense = tf.reshape(dense_flat, shape=[-1, base_dim, base_dim, base_dim, stage_layers[0]])

      print 'Decoder dense layer:', dense.get_shape().as_list()

      prev_res = dense

      for i in range(len(stage_layers)):
        next_layers = 1
        if i < len(stage_layers) - 1:
          next_layers = stage_layers[i+1]


        dims = prev_res.get_shape().as_list()

        W_i = tf.get_variable('W%d' % (i), [4, 4, 4, next_layers, stage_layers[i]])
        b_i = tf.get_variable('b%d' % (i), [next_layers])
        H_conv = tf.nn.conv3d_transpose(prev_res, W_i,
                                    [self.batch_size, 2*dims[1], 2*dims[2], 2*dims[3], next_layers],
                                    strides=[1, 2, 2, 2, 1], padding="SAME")

        H_i = tf.nn.relu(H_conv + b_i)

        print 'Decoder layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      return prev_res

  def make_x_discriminator(self, x, reuse=False):

    stage_layers = [16, 32, 64, 128, 256]

    prev_res = x
    print 'X discriminator input:', x.get_shape().as_list()

    with tf.variable_scope('X_discriminator', reuse=reuse):
      for i in range(len(stage_layers)):
        H_conv = tf.layers.conv3d(inputs=prev_res, filters=stage_layers[i], kernel_size=[4, 4, 4],
                                    padding="same", activation=tf.nn.relu)

        H_i = tf.layers.max_pooling3d(inputs=H_conv, pool_size=[2, 2, 2], strides=2)

        print 'X discriminator conv layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Now add dense layer
      last_shape = prev_res.get_shape().as_list()
      n = last_shape[1]*last_shape[2]*last_shape[3]*last_shape[4]
      flat = tf.reshape(prev_res, shape=[-1, n])
      dense = tf.layers.dense(inputs=flat, units=self.z_dim, activation=tf.nn.relu)

      print 'X discriminator dense layer:', dense.get_shape().as_list()

      # Now add decision layer
      dc = tf.layers.dense(inputs=dense, units=2)
      print 'X discriminator decision layer:', dc.get_shape().as_list()

      return dc

  def make_z_discriminator(self, z, reuse=False):

    stage_layers = [16, 32, 64]

    prev_res = z

    with tf.variable_scope('Z_discriminator', reuse=reuse):
      for i in range(len(stage_layers)):
        H_i = tf.layers.dense(inputs=prev_res, units=stage_layers[i], activation=tf.nn.relu)
        print 'Z discriminator dense layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Decision layer
      res = tf.layers.dense(inputs=prev_res, units=2)
      print 'Z discriminator decision layer:', H_i.get_shape().as_list()
      return res

  def train(self):
    # Get vars
    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'coder' in var.name]
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]

    learning_rate = 1e-3
    beta1 = 0.9
    d_optim = tf.train.AdamOptimizer().minimize(self.discriminator_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer().minimize(self.generator_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    steps = 1000

    load_time = 0
    train_time = 0
    stats_step = 10

    for step in xrange(steps):

      # Get data
      tic_load = time.time()
      batch_data = self.dm.get_full_samples(self.batch_size)
      batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim])
      toc_load = time.time()
      load_time += toc_load - tic_load

      tic_train = time.time()
      fd = {self.real_data: batch_data, self.z: batch_z}
      _, summary_d = self.sess.run([d_optim, self.discriminator_loss_summary], feed_dict = fd)
      _, summary_g = self.sess.run([g_optim, self.generator_loss_summary], feed_dict = fd)
      toc_train = time.time()
      train_time = toc_train - tic_train

      self.writer.add_summary(summary_d, step)
      self.writer.add_summary(summary_g, step)

      if step % stats_step == 0:
        print '------------------------------'
        print '  Training Step %04d' % (step)
        print ''
        print '  Waiting %5.3f sec / step for loading' % (load_time / stats_step)
        print '  Waiting %5.3f sec / step for training' % (train_time / stats_step)
        print ''

        load_time = 0
        train_time = 0

    print 'Press Enter to continue'
    raw_input()

with tf.Session() as sess:
  ali = ALI(sess, [64, 64, 64, 1], 1024)
  ali.train()
