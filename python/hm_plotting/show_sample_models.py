import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse
import pandas

nrows = 2
ncols = 2

f, axarr = plt.subplots(nrows = nrows, ncols = ncols)

for i in range(nrows):
  for j in range(ncols):
    idx = i*nrows + j

    box  = np.loadtxt('/home/aushani/summer/cc/BOX_sample_%d.csv' % idx , delimiter=',')
    star  = np.loadtxt('/home/aushani/summer/cc/STAR_sample_%d.csv' % idx, delimiter=',')

    axarr[i, j].scatter(box[:, 0], box[:, 1])
    axarr[i, j].scatter(star[:, 0], star[:, 1])
    axarr[i, j].axis("equal")
    axarr[i, j].grid(True)
    axarr[i, j].legend(['Box', 'Star'])

plt.show()
