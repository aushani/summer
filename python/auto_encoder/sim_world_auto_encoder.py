import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_manager import *

class AutoEncoder:

    def __init__(self, use_classification_loss = False):
        self.sess = tf.Session()

        self.dim_data = 31
        self.n_classes = 2

        reg_scale = 1e-7
        regularizer = tf.contrib.layers.l2_regularizer(reg_scale)

        #self.input = tf.placeholder(tf.float32, shape=[None, dim_data])
        self.input = tf.placeholder(tf.float32, shape=[None, self.dim_data, self.dim_data])
        self.label = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        flattened = tf.reshape(self.input, [-1, self.dim_data * self.dim_data])

        # Encoder
        l1 = tf.contrib.layers.fully_connected(flattened, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l2 = tf.contrib.layers.fully_connected(l1, 150,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        self.latent = tf.contrib.layers.fully_connected(l2, 2,
                activation_fn=None, weights_regularizer=regularizer)

        # Decoder
        l3 = tf.contrib.layers.fully_connected(self.latent, 150,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l4 = tf.contrib.layers.fully_connected(l3, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        self.reconstruction = tf.contrib.layers.fully_connected(l4, self.dim_data * self.dim_data,
                activation_fn=tf.nn.sigmoid, weights_regularizer=regularizer)

        # Classifier
        self.pred_label = tf.contrib.layers.fully_connected(self.latent, self.n_classes,
                activation_fn=None, weights_regularizer=regularizer)

        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.pred_label))
        reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.reconstruction, flattened))

        if use_classification_loss:
            self.loss = classification_loss + reconstruction_loss
        else:
            self.loss = reconstruction_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += reg_losses

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summaries
        self.classification_loss_summary = tf.summary.scalar('classification_loss_summary', classification_loss)
        self.reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss_summary', reconstruction_loss)
        #self.loss_summary = tf.summary.scalar('loss_summary', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy_summary', self.accuracy)

        self.summaries = tf.summary.merge_all()

    def train(self, data_manager, n_iterations=10000):
        # Set up writer
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        #iter_step = n_iterations / 100
        iter_step = 1000

        test_samples, test_labels = data_manager.test_samples, data_manager.test_labels_oh

        #for iteration in range(n_iterations):
        iteration = 0
        while True:
            sample, cn = data_manager.get_next_batch(100)
            fd = {self.input:sample, self.label:cn}

            if iteration % iter_step == 0:
                print 'Iteration %d / %d = %5.3f %%' % (iteration, n_iterations, 100.0 * iteration/n_iterations)
                print '\tOverall accuracy', self.accuracy.eval(feed_dict = {self.input: test_samples, self.label: test_labels}, session=self.sess)

                summary, _ = self.sess.run([self.summaries, self.train_step], feed_dict=fd)
                self.writer.add_summary(summary, iteration)

                ae.render_examples(dm, fn='autoencoder_examples_%08d.png' % iteration)
                ae.render_latent(dm, fn='autoencoder_latent_%08d.png' % iteration)
            else:
                self.train_step.run(feed_dict = fd, session = self.sess)

            iteration = iteration + 1

        print '\tOverall accuracy', self.accuracy.eval(feed_dict = {self.input: test_samples, self.label: test_labels}, session=self.sess)

    def render_examples(self, data_manager, n_samples=20, fn='examples.png'):
        test_samples, test_labels = data_manager.test_samples, data_manager.test_labels_oh

        reconstructed_samples = self.reconstruction.eval(feed_dict = {self.input:test_samples}, session=self.sess)
        pred_labels = self.pred_label.eval(feed_dict = {self.input:test_samples}, session=self.sess)

        plt.figure()
        for i in range(n_samples):
            im1 = test_samples[i, :, :]
            im2 = reconstructed_samples[i]
            true_label = np.argmax(test_labels[i, :])
            pred_label = np.argmax(pred_labels[i])

            #print np.min(im1), np.max(im1)
            #print np.min(im2), np.max(im2)

            im1 = np.reshape(im1, [self.dim_data, self.dim_data])
            im2 = np.reshape(im2, [self.dim_data, self.dim_data])

            plt.subplot(n_samples/4, 8, 2*i + 1)
            plt.imshow(im1)
            plt.axis('off')
            plt.title('%d' % true_label)

            plt.subplot(n_samples/4, 8, 2*i + 2)
            plt.imshow(im2)
            plt.axis('off')
            plt.title('%d' % pred_label)

        plt.savefig(fn)

    def render_latent(self, data_manager, fn='latent.png'):
        test_samples = data_manager.test_samples
        test_labels = data_manager.test_labels

        latent = self.latent.eval(feed_dict = {self.input:test_samples}, session=self.sess)

        plt.figure()
        plt.scatter(latent[:, 0], latent[:, 1], c=test_labels)
        plt.title('Latent Representation')
        plt.grid(True)

        plt.savefig(fn)

n_iterations = 1000

dm = DataManager('/home/aushani/data/auto_encoder_data/', n_test_samples = 1000)

ae = AutoEncoder(use_classification_loss=True)
ae.train(dm, n_iterations)
ae.render_examples(dm, fn='autoencoder_examples.png')
ae.render_latent(dm, fn='autoencoder_latent.png')
