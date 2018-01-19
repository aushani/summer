from auto_encoder import *

n_iterations = 10000000

ae = AutoEncoder()
ae.train(n_iterations, autoencoder=True, classifier=True, exp='both')
