from os import listdir
from os.path import isfile, join

import random
import numpy as np
import struct

import threading
import time
import Queue

class DataManager:

  def __init__(self, dirname, dim=[64, 64, 64, 1]):
    self.dirname = dirname
    self.dim = dim

    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]

    self.full_og_files = [f for f in files if 'full_' in f]

    random.shuffle(self.full_og_files)

    self.idx_at = 0

    # Start load threads
    self.data_queue = Queue.Queue(maxsize=128)

    self.load_threads = []

    for i in range(1):
      t = threading.Thread(target=self.load_files)
      t.daemon = True
      t.start()

      self.load_threads.append(t)

  def load_next_occ_grid(self):
    res = self.load_occ_grid(self.idx_at)
    self.idx_at = (self.idx_at + 1) % len(self.full_og_files)

    return res

  def load_occ_grid(self, idx):
    path = self.dirname + self.full_og_files[self.idx_at]

    fp = open(path, 'r')

    # parameters
    resolution = struct.unpack('f', fp.read(4))[0]
    num_voxels = struct.unpack('L', fp.read(8))[0]

    #print 'Have %d voxels at %5.3f m' % (num_voxels, resolution)

    # Load locations and log-likelihoods
    locs = struct.unpack('iii'*num_voxels, fp.read(4*3*num_voxels))
    lls = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Done with file
    fp.close()

    # Build grid
    grid = np.zeros(self.dim)

    # Convert to numpy
    locs = np.asarray(locs)
    lls = np.asarray(lls)

    # decompose and center
    locs_is = locs[0::3] + self.dim[0]/2
    locs_js = locs[1::3] + self.dim[1]/2
    locs_ks = locs[2::3] + self.dim[2]/2

    # find locations that are within the dimensions of our grid
    locs_is_valid = np.logical_and(locs_is >= 0, locs_is < self.dim[0])
    locs_js_valid = np.logical_and(locs_js >= 0, locs_js < self.dim[1])
    locs_ks_valid = np.logical_and(locs_ks >= 0, locs_ks < self.dim[2])

    # sample
    locs_valid = np.logical_and(locs_is_valid, np.logical_and(locs_js_valid, locs_ks_valid))
    locs_is = locs_is[locs_valid]
    locs_js = locs_js[locs_valid]
    locs_ks = locs_ks[locs_valid]
    lls = lls[locs_valid]

    # find occupied and free
    lls_pos = lls > 0
    lls_neg = lls < 0

    # subsample
    is_pos = locs_is[lls_pos]
    js_pos = locs_js[lls_pos]
    ks_pos = locs_ks[lls_pos]

    is_neg = locs_is[lls_neg]
    js_neg = locs_js[lls_neg]
    ks_neg = locs_ks[lls_neg]

    # assign in grid
    grid[is_pos, js_pos, ks_pos] = 1
    grid[is_neg, js_neg, ks_neg] = -1

    return grid

  def load_files(self):
    while True:
      self.data_queue.put(self.load_next_occ_grid())

  def get_full_samples(self, n=1):
    res = np.zeros([n] + self.dim)

    for i in range(n):
      item = self.data_queue.get()
      res[i, :, :, :, :] = item
      self.data_queue.task_done()

    return res

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  dm = DataManager('/ssd/nclt_og_data/2012-08-04/')

  start = time.time()
  grid = dm.get_full_samples(10)
  end = time.time()
  print end-start
  print grid.shape

  start = time.time()
  grid = dm.get_full_samples(10)
  end = time.time()
  print end-start
  print grid.shape

  raw_input("Press enter to plot occ grid")

  x = []
  for i in range(grid.shape[1]):
    for j in range(grid.shape[2]):
      for k in range(grid.shape[3]):
        if grid[0, i, j, k, 0] > 0:
          x.append([i, j, k])

  x = np.asarray(x)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x[:, 0], x[:, 1], -x[:, 2], c=x[:, 2])

  plt.show()

  raw_input("Press enter to quit")
