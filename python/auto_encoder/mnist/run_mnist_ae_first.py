from auto_encoder import *
import os

n_iterations = 100000

exp_name = 'ae_first'

if not os.path.exists(exp_name):
        os.makedirs(exp_name)

ae = AutoEncoder()
ae.train(n_iterations, loss='autoencoder', variables=['encoder', 'decoder'], exp_name=exp_name)
ae.train(n_iterations, loss='classifier', variables=['classifier'], exp_name=exp_name, iteration0=n_iterations)
