from os import listdir
from os.path import isfile, join

import random
import numpy as np
import struct

import threading
import time
import Queue

class DataManager:

  def __init__(self, dirname, batch_size = 1, dim=[32, 32, 32, 1]):
    self.dirname = dirname
    self.dim = dim

    self.batch_size = batch_size

    files = [f for f in listdir(dirname) if isfile(join(dirname, f))]

    self.full_og_files = [f for f in files if 'full_' in f]

    random.shuffle(self.full_og_files)

    self.idx_at = 0

    # Start load threads
    self.data_queue = Queue.Queue(maxsize=1024)

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

    # We have a dense array of voxels
    num_voxels = 32*32*32
    grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Done with file
    fp.close()

    # Build grid
    grid = np.asarray(grid_bin)
    grid = np.reshape(grid, [32, 32, 32, 1])

    return grid

  def load_files(self):
    while True:
      batch = np.empty([self.batch_size] + self.dim)
      for i in range(self.batch_size):
        batch[i, :, :, :] = self.load_next_occ_grid()

      self.data_queue.put(batch)

  def get_next_batch(self):
    item = self.data_queue.get()
    self.data_queue.task_done()

    return item

  def queue_size(self):
    return self.data_queue.qsize()

  def save(self, grid, path):
    locs = []
    lls = []

    for i in range(self.dim[0]):
      for j in range(self.dim[1]):
        for k in range(self.dim[2]):
          if abs(grid[i, j, k]) > 1e-2:
            locs.append(i-self.dim[0]/2)
            locs.append(j-self.dim[1]/2)
            locs.append(k-self.dim[2]/2)

            p = (grid[i, j, k] + 1) / 2
            if p < 0.01:
              p = 0.01
            elif p>0.99:
              p = 0.99

            ll = np.log(p/(1-p))

            lls.append(ll)

    fp = open(path, 'w')

    resolution = 1.0
    fp.write(struct.pack('f', resolution))

    num_voxels = len(lls)
    fp.write(struct.pack('L', num_voxels))

    fp.write(struct.pack('iii'*num_voxels, *locs))
    fp.write(struct.pack('f'*num_voxels, *lls))

    fp.close()

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  dm = DataManager('/ssd/nclt_og_data_dense_1m/2012-08-04/')

  start = time.time()
  grid = dm.get_next_batch()
  end = time.time()
  print end-start
  print grid.shape

  dm.save(np.squeeze(grid), 'test.sog')

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
