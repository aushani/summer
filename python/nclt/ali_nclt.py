import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_manager import *

class ALI:

  def __init__(self, sess, data_dim, z_dim):

    self.batch_size = 4

    self.use_batch_normalization = True

    #self.dm = DataManager('/ssd/nclt_og_data_dense/2012-08-04/', batch_size = self.batch_size, dim=data_dim)
    self.dm = DataManager('/ssd/nclt_og_data_dense_1m/2012-08-04/', batch_size = self.batch_size, dim=data_dim)
    #self.dm = DataManager('/home/aushani/ramdisk/', data_dim)

    self.sess = sess

    self.data_dim = data_dim
    self.z_dim = z_dim

    self.real_data = tf.placeholder(tf.float32, [None] + self.data_dim, name='real_data')
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

    # Generator
    self.encoder = self.make_encoder(self.real_data)
    self.decoder = self.make_decoder(self.z)

    # Discriminator
    self.discriminator_real = self.make_discriminator(self.real_data, self.encoder, reuse=False)
    self.discriminator_syth = self.make_discriminator(self.decoder, self.z, reuse=True)

    # Loss Functions
    self.encoder_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.discriminator_real, labels=tf.ones([self.batch_size], tf.int32)),
                            name='encoder_loss')
    self.decoder_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.discriminator_syth, labels=tf.zeros([self.batch_size], tf.int32)),
                            name='decoder_loss')

    self.generator_loss = tf.add(self.encoder_loss, self.decoder_loss, name='generator_loss')

    self.discriminator_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   logits=self.discriminator_real, labels=tf.zeros([self.batch_size], tf.int32)))
    self.discriminator_loss_syth = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   logits=self.discriminator_syth, labels=tf.ones([self.batch_size], tf.int32)))

    self.discriminator_loss = tf.add(self.discriminator_loss_real, self.discriminator_loss_syth,
                                    name='discriminator_loss')

    # Evalualation
    self.query_data = tf.placeholder(tf.float32, [None] +  self.data_dim, name='query_data')
    self.query_z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'query_z')
    self.query_logits = self.make_discriminator(self.query_data, self.query_z, reuse=True)
    self.query_softmax = tf.nn.softmax(self.query_logits)

    # Summaries
    self.encoder_loss_summary = tf.summary.scalar('encoder_loss_summary', self.encoder_loss)
    self.decoder_loss_summary = tf.summary.scalar('decoder_loss_summary', self.decoder_loss)
    self.generator_loss_summary = tf.summary.scalar('generator_loss_summary', self.generator_loss)
    self.discriminator_real_loss_summary = tf.summary.scalar('discriminator_real_loss_summary', self.discriminator_loss_real)
    self.discriminator_syth_loss_summary = tf.summary.scalar('discriminator_syth_loss_summary', self.discriminator_loss_syth)
    self.discriminator_loss_summary = tf.summary.scalar('discriminator_loss_summary', self.discriminator_loss)

  def make_encoder(self, x):

    stage_layers = [16, 32, 64, 128, 256]

    prev_res = x

    with tf.variable_scope('Encoder'):
      for i in range(len(stage_layers)):
        H_conv = tf.layers.conv3d(inputs=prev_res, filters=stage_layers[i], kernel_size=[4, 4, 4],
                                    padding="same", activation=tf.nn.relu)

        H_mp = tf.layers.max_pooling3d(inputs=H_conv, pool_size=[2, 2, 2], strides=2)
        if self.use_batch_normalization:
          H_bn = tf.layers.batch_normalization(inputs=H_mp)
          H_i = H_bn
        else:
          H_i = H_mp

        print 'Encoder conv layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Now add dense layer
      last_shape = prev_res.get_shape().as_list()
      n = last_shape[1]*last_shape[2]*last_shape[3]*last_shape[4]
      flat = tf.reshape(prev_res, shape=[-1, n])
      fc = tf.layers.dense(inputs=flat, units=2*self.z_dim, activation=tf.nn.relu)

      if self.use_batch_normalization:
        fc_bn = tf.layers.batch_normalization(inputs=fc)
        fc = fc_bn

      fc = tf.nn.relu(fc)

      print 'Encoder fully connected layer:', fc.get_shape().as_list()

      # Output layer
      output = tf.layers.dense(inputs=fc, units=self.z_dim)
      print 'Encoder output layer:', output.get_shape().as_list()

      return output


  def make_decoder(self, z):
    dense_input_layers = 2
    stage_layers = [16, 8]
    base_dim = self.data_dim[0]/(2**len(stage_layers))

    with tf.variable_scope('Decoder'):
      # Dense input layers
      prev_res = z
      for i in range(dense_input_layers):
        dense = tf.layers.dense(inputs=prev_res, units=stage_layers[0]*base_dim*base_dim*base_dim, activation=tf.nn.relu)
        if self.use_batch_normalization:
          dense = tf.layers.batch_normalization(inputs=dense)

        print 'Decoder dense input layer %d:' % (i), dense.get_shape().as_list()


      prev_res = tf.reshape(dense, shape=[-1, base_dim, base_dim, base_dim, stage_layers[0]])
      print 'Decoder final input layer:', prev_res.get_shape().as_list()

      for i in range(len(stage_layers)-1):
        next_layers = stage_layers[i+1]

        dims = prev_res.get_shape().as_list()

        W_i = tf.get_variable('W%d' % (i), [8, 8, 8, next_layers, stage_layers[i]])
        b_i = tf.get_variable('b%d' % (i), [next_layers])
        H_conv = tf.nn.conv3d_transpose(prev_res, W_i,
                                    [self.batch_size, 2*dims[1], 2*dims[2], 2*dims[3], next_layers],
                                    strides=[1, 2, 2, 2, 1], padding="SAME")

        if self.use_batch_normalization:
          H_bn = tf.layers.batch_normalization(inputs=H_conv)
          H_i = tf.nn.relu(tf.nn.bias_add(H_bn, b_i))
        else:
          H_i = tf.nn.relu(tf.nn.bias_add(H_conv, b_i))

        print 'Decoder layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Output layer
      dims = prev_res.get_shape().as_list()

      W_i = tf.get_variable('W_out', [4, 4, 4, 1, stage_layers[-1]])
      b_i = tf.get_variable('b_out', [1])
      H_conv = tf.nn.conv3d_transpose(prev_res, W_i,
                                  [self.batch_size, self.data_dim[0], self.data_dim[1], self.data_dim[2], self.data_dim[3]],
                                  strides=[1, 2, 2, 2, 1], padding="SAME")

      if self.use_batch_normalization:
        H_bn = tf.layers.batch_normalization(inputs=H_conv)
        H_out = tf.nn.bias_add(H_bn, b_i)
      else:
        H_out = tf.nn.bias_add(H_conv, b_i)

      # for enforcing range is [-1, 1]
      H_out = 2*tf.nn.sigmoid(H_out) - 1

      print 'Decoder output:',  H_out.get_shape().as_list()

      return H_out

  def make_discriminator(self, x, z, reuse=False):

    stage2_layers = [64, 32, 16]

    with tf.variable_scope('Discriminator', reuse=reuse):

      # Append x and z for input layer
      n = self.data_dim[0]*self.data_dim[1]*self.data_dim[2]*self.data_dim[3]
      x_flat = tf.reshape(x, shape=[-1, n])
      cc = tf.concat([x_flat, z], axis=1)

      print 'Discriminator cc layer:', cc.get_shape().as_list()

      prev_res = cc

      for i in range(len(stage2_layers)):
        H_fc = tf.layers.dense(inputs=prev_res, units=stage2_layers[i], activation=tf.nn.relu)
        if self.use_batch_normalization:
          H_bn = tf.layers.batch_normalization(inputs=H_fc)
          H_i = H_bn
        else:
          H_i = H_fc

        print 'D discriminator dense layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Decision layer
      dc = tf.layers.dense(inputs=prev_res, units=2)

      print 'Discriminator decision layer:', dc.get_shape().as_list()
      return dc

  def train(self):
    # Get vars
    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'Encoder' in var.name or 'Decoder' in var.name]
    self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]

    learning_rate = 1e-3
    beta1 = 0.9
    #d_optim = tf.train.AdamOptimizer().minimize(self.discriminator_loss, var_list=self.d_vars)
    #g_optim = tf.train.AdamOptimizer().minimize(self.generator_loss, var_list=self.g_vars)
    d_optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.discriminator_loss, var_list=self.d_vars)
    g_optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.generator_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
    saver = tf.train.Saver()

    step = 0

    load_time = 0
    train_time = 0
    d_steps = 0
    g_steps = 0
    stats_step = 1000

    # For samples
    sample_size = self.batch_size
    sample_z = np.random.normal(0, 1, [sample_size, self.z_dim])

    d_sums = tf.summary.merge([self.discriminator_loss_summary, self.discriminator_real_loss_summary, self.discriminator_syth_loss_summary])
    g_sums = tf.summary.merge([self.generator_loss_summary, self.decoder_loss_summary, self.encoder_loss_summary])

    while True:
      # Get data
      tic_load = time.time()
      batch_data = self.dm.get_next_batch()
      batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim])
      fd = {self.real_data: batch_data, self.z: batch_z}
      toc_load = time.time()
      load_time += toc_load - tic_load

      # Figure out which one to train
      d_loss = self.discriminator_loss.eval(fd)
      g_loss = self.generator_loss.eval(fd)

      tic_train = time.time()
      if d_loss > g_loss:
        _, summary_d = self.sess.run([d_optim, d_sums], feed_dict = fd)
        self.writer.add_summary(summary_d, step)
        d_steps += 1
      else:
        _, summary_g = self.sess.run([g_optim, g_sums], feed_dict = fd)
        self.writer.add_summary(summary_g, step)
        g_steps += 1
      toc_train = time.time()
      train_time += toc_train - tic_train

      if step % stats_step == 0:
        print '------------------------------'
        print '  Training Step %04d' % (step)
        print ''
        print '  Waiting %5.3f sec / step for loading (%d queued)' % (load_time / stats_step, self.dm.queue_size())
        print '  Waiting %5.3f sec / step for training' % (train_time / stats_step)
        print '  %04d steps for discriminator' % (d_steps)
        print '  %04d steps for generator' % (g_steps)

        load_time = 0
        train_time = 0
        d_steps = 0
        g_steps = 0

        save_path = saver.save(sess, "./model.ckpt")

        # Query
        #query_data = self.dm.get_next_batch()
        #query_z = self.encoder.eval({self.real_data: query_data})
        #print 'real', self.query_softmax.eval({self.query_data: query_data, self.query_z: query_z})

        #query_z = sample_z
        #query_data = self.decoder.eval({self.z: query_z})
        #print 'syth', self.query_softmax.eval({self.query_data: query_data, self.query_z: query_z})

        # Generate eval
        fd_eval = {self.z: sample_z}
        sample_grids = self.decoder.eval(fd_eval)
        print '  Range of sample grids: ', np.min(sample_grids[:]), np.max(sample_grids[:])

        # Save
        for i in range(sample_size):
          g = np.squeeze(sample_grids[i, :, :, :, 0])
          self.dm.save(g, 'syth_iter_%06d_%02d.sog' % (step / stats_step, i))

        print ''
        print 'Real: ', tf.nn.softmax(self.discriminator_real).eval(fd)
        print 'Syth: ', tf.nn.softmax(self.discriminator_syth).eval(fd)
        print ''

      step += 1

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  ali = ALI(sess, [32, 32, 32, 1], 256)
  ali.train()
