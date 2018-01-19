from auto_encoder import *

n_iterations = 100000

ae = AutoEncoder()
ae.train(n_iterations, autoencoder = True, exp_name='seq')
ae.train(n_iterations, classifier = True, exp_name='seq', iteration0=n_iterations)
