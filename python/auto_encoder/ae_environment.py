import numpy as np
import struct
import matplotlib.pyplot as plt
from auto_encoder import *
import time
from grid import *

# Load
dim_grid = 399
path = '/home/aushani/data/ae_sim_worlds/SIMWORLD_000002.og'
grid = Grid(path, dim_grid)

size = 200
x0 = dim_grid/2 - size/2
x1 = x0 + size

y0 = x0
y1 = x1
samples, coords = grid.get_samples(x0, x1, y0, y1)

dim_window = 31

ae = AutoEncoder(use_classification_loss=True)
ae.restore("koopa_trained/model_00770000.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

print 'Reconstructions', sample_reconstructions.shape
print 'Labels', pred_label.shape
print 'Losses', losses.shape

# Classification Results
#p0 = pred_label[:, 0] - pred_label[:, 1]
p0 = np.argmax(pred_label, axis=1)
maglist = np.exp(np.max(pred_label, axis=1)) / np.sum(np.exp(pred_label), axis=1)
maglist[p0==0] = 0

p0 = np.reshape(p0, (x1-x0, y1-y0) )
confidence = np.reshape(maglist, (x1-x0, y1-y0) )

# Rebuild
print 'Rebuilding...'
reconstructed = np.zeros((dim_grid, dim_grid))
weights = np.zeros((dim_grid, dim_grid))
best_loss = 100*np.ones((dim_grid, dim_grid))

confidence_map = np.zeros((dim_grid, dim_grid))
confidence_map[:] = np.nan
p0_map = np.zeros((dim_grid, dim_grid))
p0_map[:] = np.nan

confidence_map[x0:x1, y0:y1] = confidence
p0_map[x0:x1, y0:y1] = p0

n_samples = samples.shape[0]
for i in range(n_samples):
    if i % 1000 == 0:
        print i, n_samples

    if np.argmax(pred_label[i, :]) == 0:
        continue

    if maglist[i] < 0.95:
        continue

    x_c, y_c = coords[i, :]
    x_c = int(x_c)
    y_c = int(y_c)

    sample_reconstruction = sample_reconstructions[i, :]

    loss = losses[i]

    for ix in range(dim_window):
        for iy in range(dim_window):
            xr = x_c - dim_window/2 + ix
            yr = y_c - dim_window/2 + iy

            if loss < best_loss[xr, yr]:
                best_loss[xr, yr] = loss
                reconstructed[xr, yr] = sample_reconstruction[ix, iy]

# Now do background
#best_loss[best_loss<100] = -1
#for i in range(n_samples):
#    if i % 1000 == 0:
#        print i, n_samples
#
#    if np.argmax(pred_label[i, :]) != 0:
#        continue
#
#    #if maglist[i] < 0.99:
#    #    continue
#
#    x_c, y_c = coords[i, :]
#    x_c = int(x_c)
#    y_c = int(y_c)
#
#    sample_reconstruction = sample_reconstructions[i, :]
#
#    loss = losses[i]
#
#    for ix in range(dim_window):
#        for iy in range(dim_window):
#            xr = x_c - dim_window/2 + ix
#            yr = y_c - dim_window/2 + iy
#
#            if loss < best_loss[xr, yr]:
#                best_loss[xr, yr] = loss
#                reconstructed[xr, yr] = sample_reconstruction[ix, iy]

#reconstructed /= weights
#reconstructed = 1 / (1 + np.exp(reconstructed))

ax1 = plt.subplot(2, 2, 1)
plt.imshow(grid.grid)
#plt.axis('off')
plt.clim(0.4, 0.6)
plt.title('Data')
plt.colorbar()
plt.xlim(x0, x1)
plt.ylim(y0, y1)

plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
#plt.subplot(2, 2, 2)
plt.imshow(reconstructed)
#plt.axis('off')
plt.clim(0, 1)
plt.colorbar()
plt.title('Reconstruction')

plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
#plt.subplot(2, 2, 3)
plt.imshow(p0_map)
#plt.axis('off')
plt.title('Classification')
plt.colorbar()

plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)
#plt.subplot(2, 2, 4)
plt.imshow(confidence_map)
#plt.axis('off')
plt.colorbar()
plt.title('Confidence')
plt.clim(0.8, 1.0)

#plt.imshow(np.log10(losses_map))
#plt.axis('off')
#plt.colorbar()
#plt.clim(-5, -2)
#plt.title('Losses')

plt.show()
