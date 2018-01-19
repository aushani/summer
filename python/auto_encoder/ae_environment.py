import numpy as np
import struct
import matplotlib.pyplot as plt
from auto_encoder import *
import time

# Load
def load_sample(path):
    dim_data = 199

    fp = open(path, 'r')

    # We have a dense array of voxels
    num_voxels = dim_data*dim_data
    grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Done with file
    fp.close()

    grid = np.asarray(grid_bin)
    grid = np.reshape(grid, [dim_data, dim_data])

    return grid

grid = load_sample('/home/aushani/data/ae_sim_worlds/SIMWORLD_000002.og')
#grid = load_sample('/home/aushani/data/ae_sim_worlds/SIMWORLD_000006.og')

dim_data = 199
dim_window = 31
dim_input = 16*3*2 - 1

n_samples = (dim_data - dim_input)**2

samples = np.zeros((n_samples, dim_input, dim_input))
coords = np.zeros((n_samples, 2))
i = 0

for x_at in range(0, dim_data - dim_input):
    for y_at in range(0, dim_data - dim_input):
        sub_grid = grid[x_at:x_at + dim_input, y_at:y_at+dim_input]
        samples[i, :, :] = sub_grid
        coords[i, :] = (x_at, y_at)
        i = i + 1

ae = AutoEncoder(use_classification_loss=True)
#ae.restore("koopa_trained/model.ckpt")
ae.restore("dk_blast_trained/model_00970000.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

print 'Reconstructions', sample_reconstructions.shape
print 'Labels', pred_label.shape
print 'Losses', losses.shape

losses_map = np.reshape(losses, (dim_data-dim_input, dim_data-dim_input))

# Classification Results
#p0 = pred_label[:, 0] - pred_label[:, 1]
p0 = np.argmax(pred_label, axis=1)
maglist = np.exp(np.max(pred_label, axis=1)) / np.sum(np.exp(pred_label), axis=1)
maglist[p0==0] = 0

p0 = np.reshape(p0, [dim_data-dim_input, dim_data-dim_input])
confidence = np.reshape(maglist, [dim_data-dim_input, dim_data-dim_input])

# Rebuild
print 'Rebuilding...'
reconstructed = np.zeros((dim_data, dim_data))
weights = np.zeros((dim_data, dim_data))
best_loss = 100*np.ones((dim_data, dim_data))
for i in range(n_samples):
    if i % 1000 == 0:
        print i, n_samples

    if np.argmax(pred_label[i, :]) == 0:
        continue

    #if maglist[i] < 0.99:
    #    continue

    x_at, y_at = coords[i, :]
    x_at = int(x_at)
    y_at = int(y_at)

    sample_reconstruction = sample_reconstructions[i, :]
    #sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    # To log odds
    #log_odds = np.log(sample_reconstruction) - np.log(1-sample_reconstruction)

    loss = losses[i]

    #weight = 1 / loss
    #weight = 1

    #reconstructed[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight * sample_reconstruction
    #reconstructed[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight * log_odds
    #weights[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight
    for ix in range(dim_window):
        for iy in range(dim_window):
            xr = x_at + dim_window + ix
            yr = y_at + dim_window + iy

            if loss < best_loss[xr, yr]:
                best_loss[xr, yr] = loss
                reconstructed[xr, yr] = sample_reconstruction[ix, iy]

#reconstructed /= weights
#reconstructed = 1 / (1 + np.exp(reconstructed))

plt.subplot(1, 2, 1)
plt.imshow(grid)
#plt.axis('off')
plt.clim(0, 1)
#plt.colorbar()
plt.title('Data')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed)
#plt.axis('off')
plt.clim(0, 1)
#plt.colorbar()
plt.title('Reconstruction')

#ax1 = plt.subplot(2, 2, 1)
#plt.imshow(grid)
#plt.axis('off')
#plt.clim(0.4, 0.6)
#plt.title('Data')
#plt.colorbar()
#
#plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
#plt.imshow(reconstructed)
#plt.axis('off')
#plt.clim(0, 1)
#plt.colorbar()
#plt.title('Reconstruction')
#
#plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
#plt.imshow(p0)
#plt.axis('off')
#plt.title('Classification')
#plt.colorbar()
#
#plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)
#plt.imshow(confidence)
#plt.axis('off')
#plt.colorbar()
#plt.title('Confidence')
#plt.clim(0.0, 1.0)

#plt.imshow(np.log10(losses_map))
#plt.axis('off')
#plt.colorbar()
#plt.clim(-5, -2)
#plt.title('Losses')

plt.show()
