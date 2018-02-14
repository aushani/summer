from auto_encoder import *
import os

n_iterations = 100000

exp_name = 'classifier_first'

if not os.path.exists(exp_name):
        os.makedirs(exp_name)

ae = AutoEncoder()
ae.train(n_iterations, loss='classifier', variables=['encoder', 'classifier'], exp_name=exp_name)
ae.train(n_iterations, loss='autoencoder', variables=['decoder'], exp_name=exp_name, iteration0=n_iterations)
