import struct
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_occ_grid(grid):

    x = []
    for i in range(grid.shape[0]):
      for j in range(grid.shape[1]):
        for k in range(grid.shape[2]):
          if grid[i, j, k] > 0:
            x.append([i, j, k])

    x = np.asarray(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], -x[:, 2], c=x[:, 2])

    plt.show()


def load_occ_grid(path, dim=[128, 128, 128, 1]):

    fp = open(path, 'r')

    # parameters
    resolution = struct.unpack('f', fp.read(4))[0]
    num_voxels = struct.unpack('L', fp.read(8))[0]

    #print 'Have %d voxels at %5.3f m' % (num_voxels, resolution)

    # Load locations and log-likelihoods
    locs = struct.unpack('iii'*num_voxels, fp.read(4*3*num_voxels))
    lls = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Build grid
    grid = np.zeros(dim)
    for n in range(num_voxels):
      i = locs[3*n + 0] + dim[0]/2
      j = locs[3*n + 1] + dim[1]/2
      k = locs[3*n + 2] + dim[2]/2

      if i<0 or i>=dim[0]:
        continue

      if j<0 or j>=dim[1]:
        continue

      if k<0 or k>=dim[2]:
        continue

      if lls[n] > 0:
        grid[i, j, k] = 1
      elif lls[n] < 0:
        grid[i, j, k] = -1

    fp.close()

    return grid

start = time.time()
grid = load_occ_grid('/ssd/nclt_og_data/2012-08-04/full_1344081605386030.sog')
end = time.time()
print 'Took %5.3f sec to load' % (end - start)

#plot_occ_grid(grid)
