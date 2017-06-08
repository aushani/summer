import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_manager import *

class Direct:

    def __init__(self, sess, batch_size=1):

        self.batch_size = batch_size

        self.dm = DataManager('/home/aushani/data/shapenet_dim32_df', '/home/aushani/data/shapenet_dim32_sdf', batch_size = batch_size)

        self.sess = sess

        self.sdf = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 32, 2], name='sdf')
        self.df = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 32, 1], name='df')

        self.epn = self.make_epn(self.sdf)

        # Syntheic shape (only fill in unknown)
        sdf_known = tf.slice(self.sdf, [0, 0, 0, 0, 0], [self.batch_size, 32, 32, 32, 1])
        sdf_sign = tf.slice(self.sdf, [0, 0, 0, 0, 1], [self.batch_size, 32, 32, 32, 1])
        self.syn = tf.where(tf.greater_equal(sdf_sign, 0), sdf_known, self.epn)

        # Loss
        abs_err = tf.abs(tf.subtract(self.syn, self.df))
        self.loss = tf.reduce_sum(abs_err)/(self.batch_size*32*32*32)

        # Summaries
        self.loss_summary = tf.summary.scalar('loss_summary', self.loss)

    def make_epn(self, data):

        nf = 40
        layers = 5

        # encoder network
        last = data
        enc = [None,]*layers
        for i in range(layers):
            enc[i] = tf.layers.conv3d(inputs=last, filters=(2**i)*nf, kernel_size=[4, 4, 4],
                       padding="same", activation=tf.nn.relu)
            enc[i] = tf.layers.max_pooling3d(inputs=enc[i], pool_size=[2, 2, 2], strides=2)
            enc[i] = tf.layers.batch_normalization(enc[i])
            print 'Enc %d :' % (i), enc[i].get_shape().as_list()

            last = enc[i]

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
            # skip connection
            last = tf.concat([last, enc[-i-1]], 4)
            print ' With skip :', last.get_shape().as_list()

            output_size = int(2**(i+1))
            this_layers = last.get_shape().as_list()[-1]
            next_layers = (2**(layers-i-2))*nf
            if i is layers-1:
                next_layers = 1
            W = tf.get_variable('W_%d' % (i), [4, 4, 4, next_layers, this_layers])
            b = tf.get_variable('b_%d' % (i), [next_layers])
            dec = tf.nn.conv3d_transpose(last, W,
                    [self.batch_size, output_size, output_size, output_size, next_layers], strides=[1, 2, 2, 2, 1],
                    padding="SAME")
            dec = tf.nn.bias_add(dec, b)
            if i < layers-1:
                dec = tf.nn.relu(dec)
                dec = tf.layers.batch_normalization(dec)
            print 'Dec %d: ' % (i), dec.get_shape().as_list()

            last = dec

        res = tf.log(1 + tf.abs(last))
        print 'Res: ', res.get_shape().as_list()

        return res

    def train(self):
        # Get vars
        t_vars = tf.trainable_variables()

        learning_rate = 0.001
        beta1 = 0.9
        optim = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1).minimize(self.loss, var_list = t_vars)
        #optim = tf.train.AdamOptimizer().minimize(self.loss)
        #optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss, var_list=t_vars)
        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        saver = tf.train.Saver()

        step = 0

        load_time = 0
        train_time = 0
        step_last = step
        stats_step_min = 15
        stats_count = 0

        sums = tf.summary.merge([self.loss_summary])

        tic_last = 0

        while True:
            # Get data
            tic_load = time.time()
            data = self.dm.get_next_batch()
            fd = {self.sdf: data[0], self.df: data[1]}
            toc_load = time.time()
            load_time += toc_load - tic_load

            tic_train = time.time()
            _, summary = self.sess.run([optim, sums], feed_dict = fd)
            self.writer.add_summary(summary, step)
            toc_train = time.time()
            train_time += toc_train - tic_train

            step += 1

            if (time.time() - tic_last) > stats_step_min*60:
                print '------------------------------'
                print '  Training Step %06d' % (step)
                print ''
                print '  Waiting %5.3f sec / step for loading (%d queued)' % (load_time / (step - step_last), self.dm.queue_size())
                print '  Waiting %5.3f sec / step for training' % (train_time / (step - step_last))

                load_time = 0
                train_time = 0
                tic_last = time.time()
                step_last = step
                stats_count += 1

                save_path = saver.save(sess, "./model.ckpt")

                # Try making a dummy result
                sdfs = data[0]
                dfs = data[1]
                dfs_raw = self.epn.eval(feed_dict = {self.sdf: sdfs})
                dfs_gen = self.syn.eval(feed_dict = {self.sdf: sdfs})

                # verify
                tic_verify = time.time()
                count_known = 0
                count_unknown = 0
                for batch_i in range(self.batch_size):
                    for i in range(32):
                      for j in range(32):
                        for k in range(32):
                          if sdfs[batch_i, i, j, k, 1] >= 0:
                            if np.abs(sdfs[batch_i, i, j, k, 0] - dfs_gen[batch_i, i, j, k, 0])>1e-2:
                              count_known += 1
                          else:
                            if np.abs(dfs_raw[batch_i, i, j, k, 0] - dfs_gen[batch_i, i, j, k, 0])>1e-2:
                              count_unknown += 1
                toc_verify = time.time()
                print '  Verification: %d known mismatches, %d unknown mismatches' % (count_known, count_unknown)
                print '  Verification took %5.3f sec' % (toc_verify - tic_verify)

                for batch_i in range(self.batch_size):
                    self.dm.save(sdfs[batch_i, :, :, :, 0],     'results/shape_%02d_%05d.sdf'   % (batch_i, stats_count))
                    self.dm.save(dfs[batch_i, :, :, :, 0],      'results/shape_%02d_%05d.df'    % (batch_i, stats_count))
                    self.dm.save(dfs_raw[batch_i, :, :, :, 0],  'results/shape_%02d_%05d.dfraw' % (batch_i, stats_count))
                    self.dm.save(dfs_gen[batch_i, :, :, :, 0],  'results/shape_%02d_%05d.dfgen' % (batch_i, stats_count))

batch_size = 16

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    direct = Direct(sess, batch_size = batch_size)
    direct.train()
