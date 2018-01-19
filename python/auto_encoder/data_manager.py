import os

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import struct

import threading
import time
import Queue
import struct

class BatchMaker:

    def __init__(self, dirname, batch_size, n_test_samples=1000):
        self.dirname = dirname
        self.batch_size = batch_size

        self.dim_data = 16*3*2 - 1
        self.n_classes = 3

        self.labels = {}
        self.labels['BAC'] = 0
        self.labels['BOX'] = 1
        self.labels['STA'] = 2

        files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
        #files = [f for f in files if not 'BACKGROUND' in f]
        random.shuffle(files)

        self.test_files = files[0:n_test_samples]
        self.train_files = files[n_test_samples:]

        # Compute number of batches we'll have
        self.n_batches = len(self.train_files)/self.batch_size
        print 'Have %d batches of data' % (self.n_batches)

    def generate_data_files(self, batch_dir):
        self.batch_dir = batch_dir
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)

        print 'Generating Test Data'
        self.generate_test_data()

        print 'Generating Batch Files'
        self.make_batches()

    def generate_test_data(self):
        print 'Loading test data...'
        n_test_samples = len(self.test_files)

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

        print 'Saving test data'
        test_filename = '%s/test_samples.npy' % (self.batch_dir)
        test_labels_oh_filename = '%s/test_labels_oh.npy' % (self.batch_dir)
        test_labels_filename = '%s/test_labels.npy' % (self.batch_dir)

        np.save(test_filename, self.test_samples)
        np.save(test_labels_oh_filename, self.test_labels_oh)
        np.save(test_labels_filename, self.test_labels)
        print 'Saved test data'

    def load_filename(self, filename):
        path = self.dirname + filename
        num_voxels = self.dim_data*self.dim_data

        grid = np.fromfile(path, dtype=np.float32, count=num_voxels)
        grid = np.reshape(grid, [self.dim_data, self.dim_data])

        classname = filename[0:3]
        return grid, self.labels[classname]

    def load_idx(self, idx):
        filename = self.train_files[self.idx_at]
        return self.load_filename(filename)

    def load_next(self):
        res = self.load_idx(self.idx_at)
        self.idx_at = (self.idx_at + 1) % len(self.train_files)

        return res

    def make_next_batch(self):
        samples = np.zeros((self.batch_size, self.dim_data, self.dim_data))
        labels = np.zeros((self.batch_size, self.n_classes))

        for i in range(self.batch_size):
            sample, label = self.load_next()

            samples[i, :, :] = sample
            labels[i, label] = 1

        return samples, labels

    def make_batches(self):
        self.idx_at = 0
        self.last_batch_loaded = -1

        for batch_idx in range(0, self.n_batches):
            samples, labels = self.make_next_batch()

            samples_filename = '%s/batch_samples_%08d.npy' % (self.batch_dir, batch_idx)
            labels_filename = '%s/batch_labels_%08d.npy' % (self.batch_dir, batch_idx)

            np.save(samples_filename, samples)
            np.save(labels_filename, labels)

            self.last_batch_loaded = batch_idx


class DataManager:
    def __init__(self, batch_dir, start_at=0):
        self.batch_dir = batch_dir

        files = [f for f in os.listdir(self.batch_dir) if os.path.isfile(os.path.join(self.batch_dir, f))]
        batch_samples = [f for f in files if 'batch_samples_' in f]

        # Compute number of batches we'll have
        self.n_batches = len(batch_samples)
        print 'Have %d batches of data' % (self.n_batches)

        # Load test data
        self.load_test_data()

        # Start load threads
        self.keep_running = True
        self.data_queue = Queue.Queue(maxsize=16)

        self.batch_at = start_at % self.n_batches
        self.load_thread = threading.Thread(target=self.load_training_data)
        self.load_thread.daemon = True
        self.load_thread.start()

    def load_test_data(self):
        test_filename = '%s/test_samples.npy' % (self.batch_dir)
        test_labels_oh_filename = '%s/test_labels_oh.npy' % (self.batch_dir)
        test_labels_filename = '%s/test_labels.npy' % (self.batch_dir)

        self.test_samples = np.load(test_filename)
        self.test_labels_oh = np.load(test_labels_oh_filename)
        self.test_labels = np.load(test_labels_filename)

    def load_training_data(self):
        while self.keep_running:
            samples_filename = '%s/batch_samples_%08d.npy' % (self.batch_dir, self.batch_at)
            labels_filename = '%s/batch_labels_%08d.npy' % (self.batch_dir, self.batch_at)

            samples = np.load(samples_filename)
            labels = np.load(labels_filename)

            batch = (samples, labels)
            self.data_queue.put(batch)

            self.batch_at = (self.batch_at + 1) % (self.n_batches)

    def get_next_batch(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item

    def queue_size(self):
        return self.data_queue.qsize()

    def done(self):
        self.keep_running = False

if __name__ == '__main__':
    dm = DataManager('/home/aushani/data/batches/')

    nrows = 3
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
        plt.axis('off')
        plt.imshow(grid)
        plt.title(cn)
        plt.clim(0, 1)
        plt.colorbar()

    plt.show()

    dm.done()
