from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import struct

import threading
import time
import Queue
import struct

class DataManager:

    def __init__(self, dirname, batch_size, n_test_samples=0):
        self.dirname = dirname
        self.batch_size = batch_size

        self.dim_data = 31
        self.n_classes = 3

        self.labels = {}
        self.labels['BAC'] = 0
        self.labels['BOX'] = 1
        self.labels['STA'] = 2

        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        #files = [f for f in files if not 'BACKGROUND' in f]
        random.shuffle(files)

        self.test_files = files[0:n_test_samples]
        self.train_files = files[n_test_samples:]

        self.idx_at = 0

        # Load test
        print 'Loading test data...'
        self.test_samples = np.zeros((n_test_samples, self.dim_data, self.dim_data))
        self.test_labels_oh = np.zeros((n_test_samples, self.n_classes))
        self.test_labels = np.zeros((n_test_samples))
        for i in range(n_test_samples):
            filename = self.test_files[i]
            sample, label = self.load_filename(filename)
            self.test_samples[i, :, :] = sample
            self.test_labels_oh[i, label] = 1 #one hot
            self.test_labels[i] = label
        print 'Loaded test data'
        print 'Class statistics = ', np.sum(self.test_labels_oh, axis=0)

        # Start load threads
        self.keep_running = True
        self.data_queue = Queue.Queue(maxsize=1024)

        self.load_threads = []

        for i in range(1):
            t = threading.Thread(target=self.load_train_data)
            t.daemon = True
            t.start()

            self.load_threads.append(t)

    def load_next(self):
        res = self.load_idx(self.idx_at)
        self.idx_at = (self.idx_at + 1) % len(self.train_files)

        #res[res<0.5] = 0
        #res[res>0.5] = 1

        return res

    def load_filename(self, filename):
        #df = pd.read_csv(path, sep=',', header=None)
        path = self.dirname + filename

        fp = open(path, 'r')

        # We have a dense array of voxels
        num_voxels = self.dim_data*self.dim_data
        grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

        # Done with file
        fp.close()

        grid = np.asarray(grid_bin)
        grid = np.reshape(grid, [self.dim_data, self.dim_data])

        classname = filename[0:3]
        return grid, self.labels[classname]

    def load_idx(self, idx):
        filename = self.train_files[self.idx_at]
        return self.load_filename(filename)

    def load_train_data(self):
        while self.keep_running:
            #tic = time.time()
            samples = np.zeros((self.batch_size, self.dim_data, self.dim_data))
            labels = np.zeros((self.batch_size, self.n_classes))

            for i in range(self.batch_size):
                sample, label = self.load_next()

                samples[i, :, :] = sample
                labels[i, label] = 1

            batch = (samples, labels)
            #toc = time.time()
            #print 'took %f ms to load batch' % ((toc-tic)*1000)
            self.data_queue.put(batch)

    def get_next(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item

    def get_next_batch(self):
        return self.get_next()

    def queue_size(self):
        return self.data_queue.qsize()

    def done(self):
        self.keep_running = False

if __name__ == '__main__':
    dm = DataManager('/home/aushani/data/auto_encoder_data_bin/', batch_size=100, n_test_samples=100)

    nrows = 4
    ncols = 3

    for sp in range(nrows*ncols):
        start = time.time()
        grids, cns = dm.get_next_batch()
        end = time.time()
        print '%f msec to load' % ((end-start) * 1000)

        grid = grids[0]
        cn = np.argmax(cns[0])
        #print grid.shape

        plt.subplot(nrows, ncols, sp+1)
        plt.imshow(grid)
        plt.title(cn)
        plt.clim(0, 1)
        plt.colorbar()

    plt.show()

    dm.done()
