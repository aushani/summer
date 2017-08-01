import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse
import pandas

nrows = 2
ncols = 2

f, axarr_box_ind = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)
f, axarr_box_dep = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)

f, axarr_star_ind = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)
f, axarr_star_dep = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)

for i in range(nrows):
  for j in range(ncols):
    idx = i*nrows + j

    box_ind  = np.loadtxt('/home/aushani/summer/cc/BOX_sample_%d.csv' % idx , delimiter=',')
    box_dep  = np.loadtxt('/home/aushani/summer/cc/BOX_sample_dependent_%d.csv' % idx , delimiter=',')

    star_ind  = np.loadtxt('/home/aushani/summer/cc/STAR_sample_%d.csv' % idx , delimiter=',')
    star_dep  = np.loadtxt('/home/aushani/summer/cc/STAR_sample_dependent_%d.csv' % idx , delimiter=',')

    ax = [axarr_box_ind[i, j], axarr_box_dep[i, j], axarr_star_ind[i, j], axarr_star_dep[i, j]]
    data = [box_ind, box_dep, star_ind, star_dep]

    for a, d in zip(ax, data):
      sc = a.scatter(d[:, 0], d[:, 1], c=d[:, 2])
      plt.colorbar(sc, ax = a)
      a.axis("equal")
      a.axis((-5, 5, 10, 20))
      a.grid(True)

plt.show()
