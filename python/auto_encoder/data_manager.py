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

class DataManager:

    def __init__(self, dirname):
        self.dirname = dirname

        self.dim_data = 31
        self.n_classes = 2

        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]

        #self.files = [f for f in files if 'og' in f]
        self.files = [f for f in files if not 'BACKGROUND' in f]

        self.labels = {}
        self.labels['BOX'] = 0
        self.labels['STA'] = 1

        random.shuffle(self.files)

        self.idx_at = 0

        # Start load threads
        self.keep_running = True
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

        #res[res<0.5] = 0
        #res[res>0.5] = 1

        return res

    def load(self, idx):
        path = self.dirname + self.files[self.idx_at]
        df = pd.read_csv(path, sep=',', header=None)

        classname = self.files[self.idx_at][0:3]

        return df.values, self.labels[classname]

    def load_files(self):
        while self.keep_running:
            data = self.load_next()
            self.data_queue.put(data)

    def get_next(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item

    def get_next_batch(self, n):
        samples = np.zeros((n, self.dim_data, self.dim_data))
        labels = np.zeros((n, self.n_classes))

        for i in range(n):
            sample, label = self.get_next()

            samples[i, :, :] = sample
            labels[i, label] = 1

        return samples, labels

    def queue_size(self):
        return self.data_queue.qsize()

    def done(self):
        self.keep_running = False

if __name__ == '__main__':
    dm = DataManager('/home/aushani/data/auto_encoder_data/')

    nrows = 4
    ncols = 3

    for sp in range(nrows*ncols):
        start = time.time()
        grid, cn = dm.get_next()
        end = time.time()
        print end-start
        print grid.shape

        plt.subplot(nrows, ncols, sp+1)
        plt.imshow(grid)
        plt.title(cn)
        plt.clim(0, 1)
        plt.colorbar()

    plt.show()

    dm.done()
