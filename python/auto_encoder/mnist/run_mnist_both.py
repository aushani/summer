from auto_encoder import *

n_iterations = 10000

ae = AutoEncoder()
ae.train(n_iterations, autoencoder=True, classifier=True, exp_name='both')
