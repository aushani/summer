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

#grid = load_sample('/home/aushani/data/ae_sim_worlds/SIMWORLD_000002.og')
grid = load_sample('/home/aushani/data/ae_sim_worlds/SIMWORLD_000006.og')

dim_data = 199
dim_window = 31

n_samples = (dim_data - dim_window)**2

samples = np.zeros((n_samples, dim_window, dim_window))
coords = np.zeros((n_samples, 2))
i = 0

for x_at in range(0, dim_data - dim_window):
    for y_at in range(0, dim_data - dim_window):
        sub_grid = grid[x_at:x_at + dim_window, y_at:y_at+dim_window]
        samples[i, :, :] = sub_grid
        coords[i, :] = (x_at, y_at)
        i = i + 1

ae = AutoEncoder(use_classification_loss=True)
#ae.restore("koopa_trained/model.ckpt")
ae.restore("model_02520000.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

losses_map = np.reshape(losses, (dim_data-dim_window, dim_data-dim_window))

# Classification Results
#p0 = pred_label[:, 0] - pred_label[:, 1]
p0 = np.argmax(pred_label, axis=1)
maglist = np.exp(np.max(pred_label, axis=1)) / np.sum(np.exp(pred_label), axis=1)
maglist[p0==0] = 0

p0 = np.reshape(p0, [dim_data-dim_window, dim_data-dim_window])
confidence = np.reshape(maglist, [dim_data-dim_window, dim_data-dim_window])

frac = 1.0
a = (1 - frac)/2
b = (1 + frac)/2

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

    if maglist[i] < 0.99:
        continue

    x_at, y_at = coords[i, :]
    x_at = int(x_at)
    y_at = int(y_at)

    sample_reconstruction = sample_reconstructions[i, :]
    sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    # To log odds
    #log_odds = np.log(sample_reconstruction) - np.log(1-sample_reconstruction)

    loss = losses[i]

    #weight = 1 / loss
    #weight = 1

    #reconstructed[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight * sample_reconstruction
    #reconstructed[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight * log_odds
    #weights[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight
    for xr in range(int(x_at+a*dim_window), int(x_at + b*dim_window)):
        for yr in range(int(y_at+a*dim_window), int(y_at + b*dim_window)):
    #for xr in range(x_at, x_at + dim_window):
    #    for yr in range(y_at, y_at + dim_window):
            if loss < best_loss[xr, yr]:
                best_loss[xr, yr] = loss
                reconstructed[xr, yr] = sample_reconstruction[xr-x_at, yr-y_at]

#reconstructed /= weights
#reconstructed = 1 / (1 + np.exp(reconstructed))

ax1 = plt.subplot(2, 2, 1)
plt.imshow(grid)
plt.axis('off')
plt.clim(0.4, 0.6)
plt.title('Data')
plt.colorbar()

plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
plt.imshow(reconstructed)
plt.axis('off')
plt.clim(0, 1)
plt.colorbar()
plt.title('Reconstruction')

plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
plt.imshow(p0)
plt.axis('off')
plt.title('Classification')
plt.colorbar()

plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)
plt.imshow(confidence)
plt.axis('off')
plt.colorbar()
plt.title('Confidence')
plt.clim(0.0, 1.0)

#plt.imshow(np.log10(losses_map))
#plt.axis('off')
#plt.colorbar()
#plt.clim(-5, -2)
#plt.title('Losses')

plt.show()
