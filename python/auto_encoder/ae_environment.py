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
ae.restore("koopa_trained/model.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

losses_map = np.reshape(losses, (dim_data-dim_window, dim_data-dim_window))

# Classification Results
p0 = pred_label[:, 0] - pred_label[:, 1]
p0 = np.reshape(p0, [dim_data-dim_window, dim_data-dim_window])

# Rebuild
reconstructed = np.zeros((dim_data, dim_data))
weights = np.zeros((dim_data, dim_data))
for i in range(n_samples):
    x_at, y_at = coords[i, :]
    x_at = int(x_at)
    y_at = int(y_at)

    sample_reconstruction = sample_reconstructions[i, :]
    sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    loss = losses[i]

    weight = 1 / loss

    reconstructed[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight * sample_reconstruction
    weights[x_at:x_at + dim_window, y_at:y_at + dim_window] += weight

reconstructed /= weights

plt.subplot(1, 4, 1)
plt.imshow(grid)
plt.axis('off')
plt.clim(0, 1)
plt.title('Data')

plt.subplot(1, 4, 2)
plt.imshow(reconstructed)
plt.axis('off')
plt.clim(0, 1)
plt.title('Reconstruction')

plt.subplot(1, 4, 3)
plt.imshow(p0)
plt.clim(-5, 5)
plt.title('Classification')

plt.subplot(1, 4, 4)
plt.imshow(np.log(losses_map))
plt.axis('off')
plt.colorbar()
plt.title('Losses')

plt.show()
