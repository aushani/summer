from auto_encoder import *
from data_manager import *
import matplotlib.pyplot as plt

plt.switch_backend('agg')

last_iter = 780000
iteration_start = last_iter + 1

batch_size = 100
dm = DataManager('/home/aushani/data/batches/', start_at = iteration_start)

ae = AutoEncoder(use_classification_loss=True)
ae.restore('model_%08d.ckpt' % (last_iter))

ae.train(dm, iteration=iteration_start)
ae.render_examples(dm, fn='autoencoder_examples.png')
ae.render_latent(dm, fn='autoencoder_latent.png')
