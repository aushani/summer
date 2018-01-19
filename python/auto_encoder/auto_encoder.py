import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_manager import *
import time

class AutoEncoder:
    def __init__(self, use_classification_loss = False):
        self.sess = tf.Session()

        # Size of data we're given
        self.dim_data = 16*3*2 - 1

        # Size of data we care about reconstructing
        self.dim_window = 31

        # Classes (BOX, STAR, BACKGROUND)
        self.n_classes = 3

        self.dim_latent = 10

        reg_scale = 1e-7
        regularizer = tf.contrib.layers.l2_regularizer(reg_scale)

        #self.input = tf.placeholder(tf.float32, shape=[None, dim_data])
        self.input = tf.placeholder(tf.float32, shape=[None, self.dim_data, self.dim_data])
        self.label = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        flattened = tf.reshape(self.input, [-1, self.dim_data * self.dim_data])

        # Get the window we're interested in
        x0 = self.dim_data/2 - self.dim_window/2
        x1 = x0 + self.dim_window
        window = self.input[:, x0:x1, x0:x1]

        # Encoder
        l1 = tf.contrib.layers.fully_connected(flattened, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l2 = tf.contrib.layers.fully_connected(l1, 150,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        self.latent = tf.contrib.layers.fully_connected(l2, self.dim_latent,
                activation_fn=None, weights_regularizer=regularizer)

        # Decoder
        l3 = tf.contrib.layers.fully_connected(self.latent, 150,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l4 = tf.contrib.layers.fully_connected(l3, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        output = tf.contrib.layers.fully_connected(l4, self.dim_window * self.dim_window,
                activation_fn=tf.nn.sigmoid, weights_regularizer=regularizer)

        self.reconstruction = tf.reshape(output, [-1, self.dim_window, self.dim_window])

        # Classifier
        self.pred_label = tf.contrib.layers.fully_connected(self.latent, self.n_classes,
                activation_fn=None, weights_regularizer=regularizer)

        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.pred_label))

        # simple reconstruction loss
        #self.reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.reconstruction, flattened))

        # Compute difference, but weight unknown less
        diff = tf.squared_difference(self.reconstruction, window)
        weight = tf.abs(window - 0.5) * 2
        cost = weight * diff

        self.sample_reconstruction_loss = tf.reduce_mean(cost, axis=1)
        reconstruction_loss = tf.reduce_mean(self.sample_reconstruction_loss)

        # Loss according to gen. and dis. vox modeling by brock, lim, ritchie, weston
        #target = flattened * 3 - 1
        #output = reconstruction * 0.9 + 0.1
        #gamma = 0.97
        #reconstruction_loss = tf.reduce_mean(-gamma * target*tf.log(output) - (1-gamma) * (1-target)*tf.log(1-output))

        if use_classification_loss:
            self.loss = classification_loss + 1e3 * reconstruction_loss
        else:
            self.loss = reconstruction_loss

        reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss += reg_losses

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summaries
        self.classification_loss_summary = tf.summary.scalar('classification_loss_summary', classification_loss)
        self.reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss_summary', reconstruction_loss)
        self.loss_summary = tf.summary.scalar('loss_summary', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy_summary', self.accuracy)

        self.summaries = tf.summary.merge_all()

    def train(self, data_manager, iteration=0):
        # Set up writer
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver()

        if iteration==0:
            self.sess.run(tf.global_variables_initializer())

        #iter_step = n_iterations / 100
        iter_summaries = 1000
        iter_plots = 10000

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

        plt.figure()
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

    def reconstruct_and_classify_2(self, samples):
        n = samples.shape[0]
        reconstructions = np.zeros((n, self.dim_window*self.dim_window))
        pred_labels = np.zeros((n, self.n_classes))
        losses = np.zeros((n))

        for i, sample in enumerate(samples):
            reconstructed, pred_label, loss =  self.reconstruct_and_classify(sample)

            reconstructions[i, :] = reconstructed
            pred_labels[i, :] = pred_label
            losses[i] = loss

            print i, n

        return reconstructions, pred_labels, losses

    def render_latent(self, data_manager, fn='latent.png'):
        if self.dim_latent != 2:
            return

        test_samples = data_manager.test_samples
        test_labels = data_manager.test_labels

        latent = self.latent.eval(feed_dict = {self.input:test_samples}, session=self.sess)

        plt.figure()
        plt.scatter(latent[:, 0], latent[:, 1], c=test_labels)
        plt.title('Latent Representation')
        plt.grid(True)

        plt.savefig(fn)
