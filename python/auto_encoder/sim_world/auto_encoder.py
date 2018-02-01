import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_manager import *
import time

class AutoEncoder:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        # Size of data we're given
        self.dim_data = 16*3*2 - 1

        # Size of data we care about reconstructing
        self.dim_window = 31

        # Classes (BOX, STAR, BACKGROUND)
        self.n_classes = 3

        self.dim_latent = 10

        reg_scale = 1e-7
        self.regularizer = tf.contrib.layers.l2_regularizer(reg_scale)

        #self.input = tf.placeholder(tf.float32, shape=[None, dim_data])
        self.input = tf.placeholder(tf.float32, shape=[None, self.dim_data, self.dim_data])
        self.label = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        # Get the window we're interested in
        x0 = self.dim_data/2 - self.dim_window/2
        x1 = x0 + self.dim_window
        window = self.input[:, x0:x1, x0:x1]

        # Encoder
        self.make_encoder()

        # Decoder
        self.make_decoder()

        # Classifier
        self.make_classifier()

        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.pred_label))

        # Compute cost
        # Weight evenly between known and unknown space
        zero_cost = tf.zeros_like(self.reconstruction)

        free_mask = tf.less(window, 0.5)
        occu_mask = tf.greater(window, 0.5)

        n_free = tf.reduce_sum(tf.reduce_sum(tf.cast(free_mask, tf.float32), axis=1), axis=1)
        n_occu = tf.reduce_sum(tf.reduce_sum(tf.cast(occu_mask, tf.float32), axis=1), axis=1)

        self.cost_free = tf.where(free_mask, self.reconstruction, zero_cost)
        self.cost_occu = tf.where(occu_mask, 1-self.reconstruction, zero_cost)

        # add a tiny something to denominator in case it's 0
        self.cost_free = tf.reduce_sum(tf.reduce_sum(self.cost_free, axis=1), axis=1) / (n_free + 1e-20)
        self.cost_occu = tf.reduce_sum(tf.reduce_sum(self.cost_occu, axis=1), axis=1) / (n_occu + 1e-20)

        self.sample_reconstruction_loss = (self.cost_free + self.cost_occu) / 2

        # Mean over batch
        reconstruction_loss = tf.reduce_mean(self.sample_reconstruction_loss)

        # Some cost for confidence (ie, try to say unknown if truly unknown)
        sample_conf_cost = tf.reduce_sum(tf.reduce_sum(tf.abs(self.reconstruction - 0.5), axis=1), axis=1)
        conf_cost = tf.reduce_mean(sample_conf_cost)

        # Loss
        self.loss = classification_loss + reconstruction_loss + 1e-4 * conf_cost
        reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #self.loss += reg_losses

        #self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.train_step = tf.train.AdagradOptimizer(1e-1).minimize(self.loss)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summaries
        self.classification_loss_summary = tf.summary.scalar('classification_loss_summary', classification_loss)
        self.reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss_summary', reconstruction_loss)
        self.confidence_loss_summary = tf.summary.scalar('confidence_loss_summary', conf_cost)
        self.loss_summary = tf.summary.scalar('loss_summary', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy_summary', self.accuracy)

        self.summaries = tf.summary.merge_all()

    def make_encoder(self):
        with tf.variable_scope("encoder"):
            single_channel = tf.expand_dims(self.input, -1)

            l1 = tf.contrib.layers.conv2d(single_channel, num_outputs=50, kernel_size=self.dim_window, stride=1,
                    padding='VALID', activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l2 = tf.contrib.layers.conv2d(l1, num_outputs=150, kernel_size=1, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l3 = tf.contrib.layers.conv2d(l2, num_outputs=self.dim_latent, kernel_size=1, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l4 = tf.contrib.layers.conv2d(l3, num_outputs=self.dim_latent, kernel_size=2, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)
            l4 = tf.contrib.layers.max_pool2d(l4, kernel_size=2, stride=2)

            l5 = tf.contrib.layers.conv2d(l4, num_outputs=self.dim_latent, kernel_size=2, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)
            l5 = tf.contrib.layers.max_pool2d(l5, kernel_size=2, stride=2)

            l6 = tf.contrib.layers.conv2d(l5, num_outputs=self.dim_latent, kernel_size=2, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)
            l6 = tf.contrib.layers.max_pool2d(l6, kernel_size=2, stride=2)

            self.latent = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(l6), self.dim_latent,
                    activation_fn=None, weights_regularizer=self.regularizer)

            print 'Encoder'
            print 'Input  ', self.input.shape
            print 'L1     ', l1.shape
            print 'L2     ', l2.shape
            print 'L3     ', l3.shape
            print 'L4     ', l4.shape
            print 'L5     ', l5.shape
            print 'L6     ', l6.shape
            print 'Latent ', self.latent.shape

    def make_decoder(self):
        with tf.variable_scope("decoder"):
            l1 = tf.contrib.layers.fully_connected(self.latent, self.dim_latent*8*8,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l1 = tf.reshape(l1, [-1, 8, 8, self.dim_latent])

            l2 = tf.contrib.layers.conv2d_transpose(l1, num_outputs=self.dim_latent, kernel_size=1, stride=2,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l3 = tf.contrib.layers.conv2d_transpose(l2, num_outputs=self.dim_latent, kernel_size=1, stride=2,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l4 = tf.contrib.layers.conv2d_transpose(l3, num_outputs=150, kernel_size=1, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            l5 = tf.contrib.layers.conv2d_transpose(l4, num_outputs=50, kernel_size=1, stride=1,
                    activation_fn=tf.nn.tanh, weights_regularizer=self.regularizer)

            output = tf.contrib.layers.conv2d_transpose(l5, num_outputs=1, kernel_size=self.dim_window, stride=1,
                    activation_fn=tf.nn.sigmoid, weights_regularizer=self.regularizer)

            self.reconstruction = tf.squeeze(output[:, 0:31, 0:31, :], axis=3)

            print 'Decoder'
            print 'L1     ', l1.shape
            print 'L2     ', l2.shape
            print 'L3     ', l3.shape
            print 'L4     ', l4.shape
            print 'L5     ', l5.shape
            print 'Output ', output.shape
            print 'Res    ', self.reconstruction.shape

    def make_classifier(self):
        with tf.variable_scope("classifier"):
            self.pred_label = tf.contrib.layers.fully_connected(self.latent, self.n_classes,
                    activation_fn=None, weights_regularizer=self.regularizer)

    def train(self, data_manager, iteration=0):
        # Set up writer
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver()

        if iteration==0:
            self.sess.run(tf.global_variables_initializer())

        #iter_step = n_iterations / 100
        iter_summaries = 100
        iter_plots = 1000

        test_samples, test_labels = data_manager.test_samples, data_manager.test_labels_oh
        fd_test = {self.input:test_samples, self.label:test_labels}

        tic_step = time.time()
        while True:
            sample, cn = data_manager.get_next_batch()
            fd = {self.input:sample, self.label:cn}

            if iteration % iter_summaries == 0:
                tic_stats = time.time()

                toc_step = time.time()
                print '\n'
                print 'Iteration %d' % (iteration)
                print '\t%5.3f ms per iteration' % (1000.0 * (toc_step - tic_step) / iter_summaries)
                print '\t%5.3f sec per for step' % (toc_step - tic_step)
                print '\tOverall accuracy', self.accuracy.eval(feed_dict = fd_test, session=self.sess)

                # Summaries
                summary, _ = self.sess.run([self.summaries, self.accuracy], feed_dict=fd_test)
                self.writer.add_summary(summary, iteration)

                toc_stats = time.time()
                print '\tStats in %f sec' % (toc_stats - tic_stats)

                tic_step = time.time()

            if iteration % iter_plots == 0:
                tic_plots = time.time()
                self.render_examples(data_manager, fn='autoencoder_examples_%08d.png' % iteration)
                self.render_latent(data_manager, fn='autoencoder_latent_%08d.png' % iteration)
                toc_plots = time.time()
                print '\tPlots in %f sec' % (toc_plots - tic_plots)

                save_path = saver.save(self.sess, "./model_%08d.ckpt" % (iteration))

                tic_step = time.time()

            self.train_step.run(feed_dict = fd, session = self.sess)
            iteration = iteration + 1

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'Restored model from', filename

    def render_examples(self, data_manager, n_samples=30, fn='examples.png'):
        test_samples, test_labels = data_manager.test_samples, data_manager.test_labels

        reconstructed_samples = self.reconstruction.eval(feed_dict = {self.input:test_samples}, session=self.sess)
        pred_labels = self.pred_label.eval(feed_dict = {self.input:test_samples}, session=self.sess)

        plt.clf()
        for i in range(n_samples):
            im1 = test_samples[i, :, :]
            im2 = reconstructed_samples[i]
            true_label = test_labels[i]
            pred_label = np.argmax(pred_labels[i])

            #print np.min(im1), np.max(im1)
            #print np.min(im2), np.max(im2)

            im1 = np.reshape(im1, [self.dim_data, self.dim_data])
            im2 = np.reshape(im2, [self.dim_window, self.dim_window])

            plt.subplot(n_samples/5, 10, 2*i + 1)
            plt.imshow(im1)
            plt.clim(0, 1)
            plt.axis('off')
            plt.title('%d' % true_label)

            plt.subplot(n_samples/5, 10, 2*i + 2)
            plt.imshow(im2)
            plt.clim(0, 1)
            plt.axis('off')
            plt.title('%d' % pred_label)

        plt.savefig(fn)

    def reconstruct_and_classify(self, sample):
        if len(sample.shape) == 2:
            sample = np.reshape(sample, [1, self.dim_window, self.dim_window])

        fd = {self.input:sample}

        reconstructed = self.reconstruction.eval(feed_dict = fd, session=self.sess)
        pred_label = self.pred_label.eval(feed_dict = fd, session=self.sess)
        loss = self.sample_reconstruction_loss.eval(feed_dict = fd, session=self.sess)

        return reconstructed, pred_label, loss

    def reconstruct_and_classify_2(self, samples, chunk=1):
        n = samples.shape[0]
        reconstructions = np.zeros((n, self.dim_window, self.dim_window))
        pred_labels = np.zeros((n, self.n_classes))
        losses = np.zeros((n))

        for i in range(0, n, chunk):
            i0 = i
            i1 = i + chunk
            if i1 > n:
                i1 = n

            print '%d - %d / %d' % (i0, i1, n)
            sample_chunk = samples[i0:i1, :, :]

            reconstructed, pred_label, loss =  self.reconstruct_and_classify(sample_chunk)

            reconstructions[i0:i1, :, :] = reconstructed
            pred_labels[i0:i1, :] = pred_label
            losses[i0:i1] = loss

        return reconstructions, pred_labels, losses

    def render_latent(self, data_manager, fn='latent.png'):
        if self.dim_latent != 2:
            return

        test_samples = data_manager.test_samples
        test_labels = data_manager.test_labels

        latent = self.latent.eval(feed_dict = {self.input:test_samples}, session=self.sess)

        plt.clf()
        plt.scatter(latent[:, 0], latent[:, 1], c=test_labels)
        plt.title('Latent Representation')
        plt.grid(True)

        plt.savefig(fn)
