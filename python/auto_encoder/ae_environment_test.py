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

nrows = 32
ncols = 32

xc = 180
yc = 160

x0 = xc - nrows/2
x1 = x0 + nrows

y0 = yc - ncols/2
y1 = y0 + ncols

samples, coords = grid.get_samples(x0, x1, y0, y1)

ae = AutoEncoder()
#ae.restore("koopa_trained/model_01070000.ckpt")
#ae.restore("jan25/model_00550000.ckpt")
ae.restore("model_00160000.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

p0 = np.argmax(pred_label, axis=1)
maglist = np.exp(np.max(pred_label, axis=1)) / np.sum(np.exp(pred_label), axis=1)
#maglist[p0==0] = 0

p0 = np.reshape(p0, (x1-x0, y1-y0) )
confidence = np.reshape(maglist, (x1-x0, y1-y0) )

plt.figure(1)
plt.imshow(grid.grid)
plt.scatter(yc, xc, s=10, c='k', marker='x')
plt.axis('off')
plt.clim(0, 1)
plt.title('Data')

step = 4
n_samples = samples.shape[0]
for i in range(n_samples):
    sample = samples[i, :, :]

    sample_reconstruction = sample_reconstructions[i, :]
    sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    loss = losses[i]

    x, y = coords[i, :]

    if (x - x0) % step == 0 and (y - y0) % step == 0:
        ix = (x-x0)/step
        iy = (y-y0)/step
        sp = int(ix*ncols/step + iy) + 1

        plt.figure(2)
        plt.subplot(nrows/step, ncols/step, sp)

        plt.imshow(sample_reconstruction)
        plt.clim(0, 1)
        plt.axis('off')
        #plt.title('%f' % loss)

        plt.figure(3)
        ax = plt.subplot(nrows/step, ncols/step, sp)

        plt.imshow(sample)
        xr = sample.shape[0]/2 - dim_window/2
        yr = xr
        ax.add_patch(patches.Rectangle((xr, yr), dim_window, dim_window, fill=False))
        plt.clim(0, 1)
        plt.axis('off')

        plt.figure(4)
        plt.scatter(y-y0, x-x0, marker='x', c='b')

        plt.figure(5)
        plt.scatter(y-y0, x-x0, marker='x', c='b')

        plt.figure(6)
        plt.scatter(y-y0, x-x0, marker='x', c='b')

losses = np.reshape(losses, [nrows, ncols])

plt.figure(4)
plt.imshow(losses)
plt.title('Reconstruction Loss')
plt.colorbar()

plt.figure(5)
plt.imshow(confidence)
plt.title('Confidence')
plt.colorbar()

plt.figure(6)
plt.imshow(p0)
plt.title('Class')
plt.colorbar()

plt.show()
