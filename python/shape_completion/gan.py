import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_manager import *

class GAN:

  def __init__(self, sess, data_dim, z_dim):

    self.batch_size = 4

    self.use_batch_normalization = True

    self.dm = DataManager('/home/aushani/data/shapenet_dim32_df/', batch_size = self.batch_size)

    self.sess = sess

    self.data_dim = data_dim
    self.z_dim = z_dim

    self.real_data = tf.placeholder(tf.float32, [None] + self.data_dim, name='real_data')
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

    # Generator
    self.decoder = self.make_decoder(self.z)

    # Discriminator
    self.discriminator_real = self.make_discriminator(self.real_data, reuse=False)
    self.discriminator_syth = self.make_discriminator(self.decoder, reuse=True)

    # Loss Functions
    self.generator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.discriminator_syth, labels=tf.zeros([self.batch_size], tf.int32)),
                            name='generator_loss')

    self.discriminator_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   logits=self.discriminator_real, labels=tf.zeros([self.batch_size], tf.int32)))
    self.discriminator_loss_syth = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   logits=self.discriminator_syth, labels=tf.ones([self.batch_size], tf.int32)))

    self.discriminator_loss = tf.add(self.discriminator_loss_real, self.discriminator_loss_syth,
                                    name='discriminator_loss')

    # Summaries
    self.generator_loss_summary = tf.summary.scalar('generator_loss_summary', self.generator_loss)
    self.discriminator_real_loss_summary = tf.summary.scalar('discriminator_real_loss_summary', self.discriminator_loss_real)
    self.discriminator_syth_loss_summary = tf.summary.scalar('discriminator_syth_loss_summary', self.discriminator_loss_syth)
    self.discriminator_loss_summary = tf.summary.scalar('discriminator_loss_summary', self.discriminator_loss)

  def make_decoder(self, z):
    sf = 10
    stage_layers = [sf*64, sf*32, sf*16]
    base_dim = self.data_dim[0]/(2**len(stage_layers))

    with tf.variable_scope('Decoder'):
      # Dense layer
      dense_flat = tf.layers.dense(inputs=z, units=stage_layers[0]*base_dim*base_dim*base_dim)
      fc = tf.reshape(dense_flat, shape=[-1, base_dim, base_dim, base_dim, stage_layers[0]])

      if self.use_batch_normalization:
        input_layer = tf.layers.batch_normalization(inputs=fc)
      else:
        input_layer = fc

      print 'Decoder fully connected input layer:', input_layer.get_shape().as_list()

      prev_res = input_layer

      for i in range(len(stage_layers)-1):
        next_layers = stage_layers[i+1]

        dims = prev_res.get_shape().as_list()

        W_i = tf.get_variable('W%d' % (i), [4, 4, 4, next_layers, stage_layers[i]])
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

      W_i = tf.get_variable('W_out', [4, 4, 4, 2, stage_layers[-1]])
      b_i = tf.get_variable('b_out', [2])
      H_conv = tf.nn.conv3d_transpose(prev_res, W_i,
                                  [self.batch_size, self.data_dim[0], self.data_dim[1], self.data_dim[2], self.data_dim[3]],
                                  strides=[1, 2, 2, 2, 1], padding="SAME")

      if self.use_batch_normalization:
        H_bn = tf.layers.batch_normalization(inputs=H_conv)
        H_out = tf.nn.bias_add(H_bn, b_i)
      else:
        H_out = tf.nn.bias_add(H_conv, b_i)

      # for enforcing range is [-1, 1]
      #H_out = 2*tf.nn.sigmoid(H_out) - 1

      # log space
      H_out = tf.log(1 + tf.abs(H_out))

      print 'Decoder output:',  H_out.get_shape().as_list()

      return H_out

  def make_discriminator(self, x, reuse=False):

    stage2_layers = []

    with tf.variable_scope('Discriminator', reuse=reuse):

      n = self.data_dim[0]*self.data_dim[1]*self.data_dim[2]*self.data_dim[3]
      x_flat = tf.reshape(x, shape=[-1, n])
      prev_res = x_flat

      for i in range(len(stage2_layers)):
        H_fc = tf.layers.dense(inputs=prev_res, units=stage2_layers[i],
                activation=tf.nn.relu, name='H_fc_%d' % (i), reuse=reuse)
        if self.use_batch_normalization:
          H_bn = tf.layers.batch_normalization(inputs=H_fc, name='H_fc_bn_%d' % (i), reuse=reuse)
          H_i = H_bn
        else:
          H_i = H_fc

        print 'D discriminator dense layer %d:' % (i), H_i.get_shape().as_list()
        prev_res = H_i

      # Decision layer
      dc = tf.layers.dense(inputs=prev_res, units=2, name='dc', reuse=reuse)

      print 'Discriminator decision layer:', dc.get_shape().as_list()
      return dc

  def train(self):
    # Get vars
    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'Decoder' in var.name]
    self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]

    #learning_rate = 1e-5
    #beta1 = 0.9
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
    stats_step = 100

    # For samples
    sample_size = self.batch_size
    sample_z = np.random.normal(0, 1, [sample_size, self.z_dim])

    d_sums = tf.summary.merge([self.discriminator_loss_summary, self.discriminator_real_loss_summary, self.discriminator_syth_loss_summary])
    g_sums = tf.summary.merge([self.generator_loss_summary])

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

        # Generate eval
        fd_eval = {self.z: sample_z}
        sample_grids = self.decoder.eval(fd_eval)
        print '  Range of sample grids (label): ', np.min(sample_grids[:, :, :, :, 0]), np.max(sample_grids[:, :, :, :, 0])
        print '  Range of sample grids (dist.): ', np.min(sample_grids[:, :, :, :, 1]), np.max(sample_grids[:, :, :, :, 1])
        print '  Range of batch  grids (label): ', np.min(batch_data[:, :, :, :, 0]), np.max(batch_data[:, :, :, :, 0])
        print '  Range of batch  grids (dist.): ', np.min(batch_data[:, :, :, :, 1]), np.max(batch_data[:, :, :, :, 1])

        # Save
        for i in range(sample_size):
          g = np.squeeze(sample_grids[i, :, :, :, :])
          count = np.count_nonzero(np.abs(self.dm.unscale_df(g[:, :, :, 1])) < 0.5)
          print '   Sample %d has %d occupied elements' % (i, count)
          self.dm.save_dense(g, 'syth_iter_%06d_%02d.og' % (step / stats_step, i))

        print ''
        print 'Real: ', tf.nn.softmax(self.discriminator_real).eval(fd)
        print 'Syth: ', tf.nn.softmax(self.discriminator_syth).eval(fd)
        print ''


      step += 1

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  gan = GAN(sess, [32, 32, 32, 2], 640)
  gan.train()
