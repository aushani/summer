import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoder:

    def __init__(self, use_classification_loss = False):
        dim_data = 784
        n_classes = 10

        reg_scale = 1e-7
        regularizer = tf.contrib.layers.l2_regularizer(reg_scale)

        self.input = tf.placeholder(tf.float32, shape=[None, dim_data])
        self.label = tf.placeholder(tf.float32, shape=[None, n_classes])

        # Encoder
        l1 = tf.contrib.layers.fully_connected(self.input, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l2 = tf.contrib.layers.fully_connected(l1, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        self.latent = tf.contrib.layers.fully_connected(l2, 2,
                activation_fn=None, weights_regularizer=regularizer)

        # Decoder
        l3 = tf.contrib.layers.fully_connected(self.latent, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        l4 = tf.contrib.layers.fully_connected(l3, 50,
                activation_fn=tf.nn.tanh, weights_regularizer=regularizer)

        self.reconstruction = tf.contrib.layers.fully_connected(l4, dim_data,
                activation_fn=tf.nn.sigmoid, weights_regularizer=regularizer)

        # Classifier
        self.pred_label = tf.contrib.layers.fully_connected(self.latent, n_classes,
                activation_fn=None, weights_regularizer=regularizer)

        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.pred_label))
        reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.reconstruction, self.input))

        if use_classification_loss:
            self.loss = classification_loss + reconstruction_loss
        else:
            self.loss = reconstruction_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += reg_losses

        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess = tf.Session()

    def train(self, n_iterations=10000):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.sess.run(tf.global_variables_initializer())
        iter_step = n_iterations / 100
        for iteration in range(n_iterations):
            if iteration % iter_step == 0:
                print 'Iteration %d / %d = %5.3f %%' % (iteration, n_iterations, 100.0 * iteration/n_iterations)

                correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred_label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print '\tOverall accuracy', accuracy.eval(feed_dict = {self.input: mnist.test.images, self.label: mnist.test.labels}, session=self.sess)

            batch = mnist.train.next_batch(100)
            fd = {self.input:batch[0], self.label:batch[1]}
            self.train_step.run(feed_dict = fd, session=self.sess)

        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print 'Overall accuracy', accuracy.eval(feed_dict = {self.input: mnist.test.images, self.label: mnist.test.labels}, session=self.sess)

    def render_examples(self, n_images=20, fn='examples.png'):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        test_images = mnist.test.images
        test_labels = mnist.test.labels

        test_images = test_images[10:(10+n_images), :]
        test_labels = test_labels[10:(10+n_images), :]

        reconstructed_images = self.reconstruction.eval(feed_dict = {self.input:test_images}, session=self.sess)
        pred_labels = self.pred_label.eval(feed_dict = {self.input:test_images}, session=self.sess)

        plt.figure()
        for i in range(n_images):
            im1 = test_images[i]
            im2 = reconstructed_images[i]
            pred_label = np.argmax(pred_labels[i])
            true_label = np.argmax(test_labels[i])

            #print np.min(im1), np.max(im1)
            #print np.min(im2), np.max(im2)

            im1 = np.reshape(im1, [28, 28])
            im2 = np.reshape(im2, [28, 28])

            plt.subplot(n_images/4, 8, 2*i + 1)
            plt.imshow(im1)
            plt.axis('off')
            plt.title('%d' % true_label)

            plt.subplot(n_images/4, 8, 2*i + 2)
            plt.imshow(im2)
            plt.axis('off')
            plt.title('%d' % pred_label)

        plt.savefig(fn)

    def render_latent(self, fn='latent.png'):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

        test_images = mnist.test.images
        test_labels = mnist.test.labels

        latent = self.latent.eval(feed_dict = {self.input:test_images}, session=self.sess)

        print test_labels.shape
        print latent.shape

        plt.figure()
        plt.scatter(latent[:, 0], latent[:, 1], c=test_labels)
        plt.title('Latent Representation')
        plt.grid(True)

        plt.savefig(fn)

n_iterations = 100000

ae = AutoEncoder(use_classification_loss=False)
ae.train(n_iterations)
ae.render_examples(fn='autoencoder_examples.png')
ae.render_latent(fn='autoencoder_latent.png')

ae = AutoEncoder(use_classification_loss=True)
ae.train(n_iterations)
ae.render_examples(fn='autoencoder_classification_examples.png')
ae.render_latent(fn='autoencoder_classification_latent.png')
