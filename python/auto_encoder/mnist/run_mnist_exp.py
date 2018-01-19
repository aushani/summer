from auto_encoder import *

n_iterations = 1000000

ae1 = AutoEncoder()
ae1.train(n_iterations, autoencoder = True)
ae1.train(n_iterations, classifier = True)
ae1.render_examples(fn='ae_ex_seq.png')
ae1.render_latent(fn='ae_latent_seq.png')

ae2 = AutoEncoder()
ae2.train(n_iterations, autoencoder = True, classifier = True)
ae2.render_examples(fn='ae_ex_both.png')
ae2.render_latent(fn='ae_latent_both.png')
