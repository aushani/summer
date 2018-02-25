from auto_encoder import *
import os

n_iterations = 1000000

exp_name = 'ae_first'

if not os.path.exists(exp_name):
        os.makedirs(exp_name)

ae = AutoEncoder()
ae.train(n_iterations, loss='autoencoder', variables=['encoder', 'decoder'], exp_name=exp_name)
ae.train(n_iterations, loss='classifier', variables=['classifier'], exp_name=exp_name, iteration0=n_iterations)

test_images, test_labels, reconstructed_images, latent, pred_labels = ae.process_all()

np.save('%s/test_images.npy' % (exp_name), test_images)
np.save('%s/test_labels.npy' % (exp_name), test_labels)
np.save('%s/reconstructed_images.npy' % (exp_name), reconstructed_images)
np.save('%s/latent.npy' % (exp_name), latent)
np.save('%s/pred_labels.npy' % (exp_name), pred_labels)
