import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from auto_encoder import *
import time
from grid import *

# Load
dim_grid = 399
path = '/home/aushani/data/ae_sim_worlds/SIMWORLD_000002.og'
grid = Grid(path, dim_grid)

dim_window = 31

nrows = 24
ncols = 24

xc = 180
yc = 160

x0 = xc - nrows/2
x1 = x0 + nrows

y0 = yc - ncols/2
y1 = y0 + ncols

samples, coords = grid.get_samples(x0, x1, y0, y1)

ae = AutoEncoder(use_classification_loss=True)
ae.restore("koopa_trained/model_00770000.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

plt.figure(1)
plt.imshow(grid.grid)
plt.scatter(yc, xc, s=10, c='k', marker='x')
plt.axis('off')
plt.clim(0, 1)
plt.title('Data')

n_samples = samples.shape[0]
for i in range(n_samples):
    sample = samples[i, :, :]

    sample_reconstruction = sample_reconstructions[i, :]
    sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    loss = losses[i]

    x, y = coords[i, :]

    if (x - x0) % 4 == 0 and (y - y0) % 4 == 0:
        ix = (x-x0)/4
        iy = (y-y0)/4
        sp = int(ix*ncols/4 + iy) + 1

        plt.figure(2)
        plt.subplot(nrows/4, ncols/4, sp)

        plt.imshow(sample_reconstruction)
        plt.clim(0, 1)
        plt.axis('off')
        #plt.title('%f' % loss)

        plt.figure(3)
        ax = plt.subplot(nrows/4, ncols/4, sp)

        plt.imshow(sample)
        xr = sample.shape[0]/2 - dim_window/2
        yr = xr
        ax.add_patch(patches.Rectangle((xr, yr), dim_window, dim_window, fill=False))
        plt.clim(0, 1)
        plt.axis('off')

        plt.figure(4)
        plt.scatter(x-x0, y-y0, marker='x', c='b')

losses = np.reshape(losses, [nrows, ncols])

plt.figure(4)
plt.imshow(losses)
plt.title('Reconstruction Loss')

plt.show()
