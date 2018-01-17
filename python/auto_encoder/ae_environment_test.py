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

nrows = 64
ncols = 64

xc = 96
yc = 37

x0 = xc - nrows/2
x1 = x0 + nrows

y0 = yc - ncols/2
y1 = y0 + ncols

for x_at in range(x0, x1):
    for y_at in range(y0, y1):
        sub_grid = grid[x_at:x_at + dim_window, y_at:y_at+dim_window]
        samples[i, :, :] = sub_grid
        coords[i, :] = (x_at, y_at)
        i = i + 1

n_samples = i
samples = samples[:n_samples, :, :]
coords = coords[:n_samples, :]

ae = AutoEncoder(use_classification_loss=True)
#ae.restore("koopa_trained/model.ckpt")
ae.restore("koopa_trained_more/model.ckpt")

print 'Sample shape', samples.shape

tic = time.time()
sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify(samples)
#sample_reconstructions, pred_label, losses = ae.reconstruct_and_classify_2(samples)
toc = time.time()
print 'Took %5.3f ms' % ((toc - tic)*1e3)

plt.figure()
plt.imshow(grid)
plt.scatter(yc + dim_window/2, xc+dim_window/2, s=10, c='k', marker='x')
plt.axis('off')
plt.clim(0, 1)
plt.title('Data')

plt.figure()
for i in range(n_samples):
    sample_reconstruction = sample_reconstructions[i, :]
    sample_reconstruction = np.reshape(sample_reconstruction, (dim_window, dim_window))

    loss = losses[i]

    x, y = coords[i, :]

    if (x - x0) % 4 == 0 and (y - y0) % 4 == 0:
        ix = (x-x0)/4
        iy = (y-y0)/4
        sp = int(ix*ncols/4 + iy) + 1
        plt.subplot(nrows/4, ncols/4, sp)

        plt.imshow(sample_reconstruction)
        plt.clim(0, 1)
        plt.axis('off')
        #plt.title('%f' % loss)

losses = np.reshape(losses, [nrows, ncols])

plt.figure()
plt.imshow(losses)
plt.title('Reconstruction Loss')

plt.show()

