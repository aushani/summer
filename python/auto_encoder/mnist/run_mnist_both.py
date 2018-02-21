from auto_encoder import *
import os

n_iterations = 10000
exp_name = 'simultaneous'

if not os.path.exists(exp_name):
        os.makedirs(exp_name)

ae = AutoEncoder()
ae.train(n_iterations, loss='both', variables=['encoder', 'decoder', 'classifier'], exp_name=exp_name)
