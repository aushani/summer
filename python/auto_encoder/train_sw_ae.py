from auto_encoder import *
from data_manager import *
import matplotlib.pyplot as plt

plt.switch_backend('agg')

batch_size = 100
dm = DataManager('/home/aushani/data/batches/')

ae = AutoEncoder(use_classification_loss=True)
ae.train(dm)
ae.render_examples(dm, fn='autoencoder_examples.png')
ae.render_latent(dm, fn='autoencoder_latent.png')
