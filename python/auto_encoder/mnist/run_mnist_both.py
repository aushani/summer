from auto_encoder import *
import os
import argparse

parser = argparse.ArgumentParser(description='MNIST Autoencoder')
parser.add_argument('-c','--classification_weight', default=1.0, help ='The weight for classification loss.', type=float)

args = parser.parse_args()

n_iterations = 1000000
cw = args.classification_weight
exp_name = 'simultaneous_%7.5f' % (cw)

if not os.path.exists(exp_name):
        os.makedirs(exp_name)

ae = AutoEncoder()
ae.train(n_iterations, loss='both', variables=['encoder', 'decoder', 'classifier'], exp_name=exp_name)

test_images, test_labels, reconstructed_images, latent, pred_labels = ae.process_all()

np.save('%s/test_images.npy' % (exp_name), test_images)
np.save('%s/test_labels.npy' % (exp_name), test_labels)
np.save('%s/reconstructed_images.npy' % (exp_name), reconstructed_images)
np.save('%s/latent.npy' % (exp_name), latent)
np.save('%s/pred_labels.npy' % (exp_name), pred_labels)
