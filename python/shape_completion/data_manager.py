from os import listdir
from os.path import isfile, isdir, join

import random
import numpy as np
import struct

import threading
import time
import Queue

class DataManager:

  def __init__(self, dirname, batch_size = 1, dim=[32, 32, 32, 2]):
    self.dirname = dirname
    self.dim = dim

    self.batch_size = batch_size

    dirs = [f for f in listdir(dirname) if isdir(join(dirname, f))]

    files = [dirname + '/' + d + '/' + f for d in dirs for f in listdir(dirname + '/' + d) if '.df' in f]
    self.files = files

    random.shuffle(self.files)

    self.idx_at = 0

    # Start load threads
    self.data_queue = Queue.Queue(maxsize=1024)

    self.load_threads = []

    for i in range(1):
      t = threading.Thread(target=self.load_files)
      t.daemon = True
      t.start()

      self.load_threads.append(t)

  def load_next(self):
    res = self.load(self.idx_at)
    self.idx_at = (self.idx_at + 1) % len(self.files)

    return res

  def load(self, idx):
    path = self.files[self.idx_at]

    fp = open(path, 'r')

    # We have a dense array of voxels
    dim_str = fp.read(3*8)
    num_voxels = 32*32*32
    grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Done with file
    fp.close()

    # Build grid
    grid = np.asarray(grid_bin)
    grid = np.reshape(grid, [32, 32, 32], order='C')

    sample = np.zeros(self.dim)
    sample[:, :, :, 0] = (np.isfinite(grid) & (np.abs(grid) < 0.5)) * 2 - 1
    sample[:, :, :, 1] = np.abs(grid)/20

    return sample

  def load_files(self):
    while True:
      batch = np.empty([self.batch_size] + self.dim)
      for i in range(self.batch_size):
        batch[i, :, :, :, :] = self.load_next()

      self.data_queue.put(batch)

  def get_next_batch(self):
    item = self.data_queue.get()
    self.data_queue.task_done()

    return item

  def queue_size(self):
    return self.data_queue.qsize()

  def save(self, sample, path):
    locs = []
    lls = []

    for i in range(self.dim[0]):
      for j in range(self.dim[1]):
        for k in range(self.dim[2]):
          locs.append(i-self.dim[0]/2)
          locs.append(j-self.dim[1]/2)
          locs.append(k-self.dim[2]/2)

          if np.isfinite(sample[i, j, k, 1]) and abs(sample[i, j, k, 1]*20) < 0.5:
          #if sample[i, j, k, 0] > 0:
            p = 0.99
          else:
            p = 0.01

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

  def save_dense(self, sample, path):

    fp = open(path, 'w')

    for i in range(self.dim[0]):
      for j in range(self.dim[1]):
        for k in range(self.dim[2]):
          if np.abs(sample[i, j, k, 1]*20) < 0.5:
            val = 0.99
          else:
            val = 0.01

          fp.write(struct.pack('f', val))

    fp.close()

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  dm = DataManager('/ssd/shapenet_dim32_df/')

  res = dm.get_next_batch()

  print res.shape
  sample = res[0, :, :, :, :]
  print sample.shape
  dm.save(sample, 'test.sog')
