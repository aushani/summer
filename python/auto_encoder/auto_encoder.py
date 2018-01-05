import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev = 0.1)
    return tf.Variable(initial)

def fc_layer(previous, output_size):
    input_size = previous.get_shape().as_list()[1]

    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b

class AutoEncoder:

    def __init__(self):
        dim_data = 784

        self.input = tf.placeholder(tf.float32, shape=[None, dim_data])

        l1 = tf.nn.tanh(fc_layer(self.input, 50))
        l2 = tf.nn.tanh(fc_layer(l1, 50))

        latent = fc_layer(l2, 2)

        l3 = tf.nn.tanh(fc_layer(latent, 50))
        l4 = tf.nn.tanh(fc_layer(l3, 50))
        #self.reconstruction = tf.nn.tanh(fc_layer(l4, dim_data))
        self.reconstruction = fc_layer(l4, dim_data)

        self.loss = tf.reduce_mean(tf.squared_difference(self.reconstruction, self.input))

        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train(self, n_images=10):

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iteration in range(100000):
                print 'Iteration %d' % (iteration)
                batch = mnist.train.next_batch(100)
                #print 'Loss: ', self.loss.eval(feed_dict = {self.input:batch[0]})
                self.train_step.run(feed_dict = {self.input:batch[0]})
                #print 'Loss: ', self.loss.eval(feed_dict = {self.input:batch[0]})

            test_images = mnist.test.images
            test_images = test_images[10:10+n_images, :]
            reconstructed_images = self.reconstruction.eval(feed_dict = {self.input:test_images})

            for i in range(n_images):
                im1 = test_images[i]
                im2 = reconstructed_images[i]

                print np.min(im1), np.max(im1)
                print np.min(im2), np.max(im2)

                im1 = np.reshape(im1, [28, 28])
                im2 = np.reshape(im2, [28, 28])

                plt.subplot(n_images/2, 4, 2*i + 1)
                plt.imshow(im1)

                plt.subplot(n_images/2, 4, 2*i + 2)
                plt.imshow(im2)

            plt.show()

ae = AutoEncoder()
ae.train()
