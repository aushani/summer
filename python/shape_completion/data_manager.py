from os import listdir
from os.path import isfile, isdir, join, basename, splitext

import random
import numpy as np
import struct

import threading
import time
import Queue

import bisect
import re

class DataManager:

  def __init__(self, dirname_df, dirname_sdf, batch_size=1):
    self.batch_size = batch_size

    dirs_df = [f for f in listdir(dirname_df) if isdir(join(dirname_df, f))]
    files_df = [join(d, f) for d in dirs_df for f in listdir(join(dirname_df, d)) if '.df' in f]
    files_df.sort()

    dirs_sdf = [f for f in listdir(dirname_sdf) if isdir(join(dirname_sdf, f))]
    files_sdf = [join(d, f) for d in dirs_sdf for f in listdir(join(dirname_sdf, d)) if '.sdf' in f]
    files_sdf.sort()

    # Find pairs that exist in both
    self.file_pairs = {}
    for sdf in files_sdf:
      root, ext = splitext(sdf)
      df_cand = re.sub('__\d+__.sdf', '__0__.df', sdf)
      i = bisect.bisect_left(files_df, df_cand)
      if i != len(files_df) and files_df[i] == df_cand:
        self.file_pairs[join(dirname_sdf, sdf)] = join(dirname_df, df_cand)

    print 'Have %d samples' % (len(self.file_pairs))
    self.files_sdf = self.file_pairs.keys()
    random.shuffle(self.files_sdf)

    self.idx_at = 0
    self.epoch = 1

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
    self.idx_at += 1
    if self.idx_at >= len(self.file_pairs):
      self.idx_at = 0
      print 'Starting epoch %04d...' % (self.epoch)

    return res

  def read_file(self, fn):
    fp = open(fn, 'r')
    dim_str = fp.read(3*8)
    num_voxels = 32*32*32
    grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

    # Done with file
    fp.close()

    # Build grid
    grid = np.asarray(grid_bin)
    grid = np.reshape(grid, [1, 32, 32, 32], order='C')

    return grid

  def load(self, idx):
    sdf = self.files_sdf[self.idx_at]
    df = self.file_pairs[sdf]

    grid_sdf = self.read_file(sdf)
    grid_df = self.read_file(df)

    sample_sdf = np.empty([1, 32, 32, 32, 2])
    sample_df = np.empty([1, 32, 32, 32, 1])

    sample_sdf[:, :, :, :, 0] = self.scale(grid_sdf)
    sample_sdf[:, :, :, :, 1] = np.sign(grid_sdf)

    sample_df[:, :, :, :, 0] = self.scale(grid_df)

    return (sample_sdf, sample_df)

  def scale(self, df):
      df[np.isinf(df)] = np.max(df[np.isfinite(df)])
      return np.log(np.abs(df)+1.0)

  def unscale(self, df):
      return np.exp(df)-1.0

  def load_files(self):
    while True:
      samples_sdf = np.empty([self.batch_size, 32, 32, 32, 2])
      samples_df = np.empty([self.batch_size, 32, 32, 32, 1])
      for i in range(self.batch_size):
        sdf, df = self.load_next()
        samples_sdf[i, :, :, :, :] = sdf
        samples_df[i, :, :, :, :] = df

      self.data_queue.put((samples_sdf, samples_df))

  def get_next_batch(self):
    item = self.data_queue.get()
    self.data_queue.task_done()

    return item

  def queue_size(self):
    return self.data_queue.qsize()

  def save(self, grid, path):
    f = open(path, 'w')

    f.write(struct.pack('q', 32))
    f.write(struct.pack('q', 32))
    f.write(struct.pack('q', 32))

    df = self.unscale(grid)

    for i in range(32):
      for j in range(32):
        for k in range(32):
          f.write(struct.pack('f', df[i, j, k]))

    f.close()

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  dm = DataManager('/home/aushani/data/shapenet_dim32_df', '/home/aushani/data/shapenet_dim32_sdf')
  x = dm.get_next_batch()[0]
  print x.shape
  x = x[0, :, :, :, 1]
  print x.shape
  print np.sum(x<0)
  print np.sum(x>0)
