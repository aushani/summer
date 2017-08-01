import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse
import pandas

nrows = 4
ncols = 4

f, axarr = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)

for j in range(ncols):
  idx = j

  box_ind  = np.loadtxt('/home/aushani/summer/cc/BOX_sample_%d.csv' % idx , delimiter=',')
  box_dep  = np.loadtxt('/home/aushani/summer/cc/BOX_sample_dependent_%d.csv' % idx , delimiter=',')

  star_ind  = np.loadtxt('/home/aushani/summer/cc/STAR_sample_%d.csv' % idx , delimiter=',')
  star_dep  = np.loadtxt('/home/aushani/summer/cc/STAR_sample_dependent_%d.csv' % idx , delimiter=',')

  data = [box_ind, box_dep, star_ind, star_dep]

  for i in range(nrows):
    a = axarr[i, j]
    d = data[i]
    sc = a.scatter(d[:, 0], d[:, 1], c=d[:, 2])
    plt.colorbar(sc, ax = a)
    a.axis("equal")
    a.axis((-5, 5, 10, 20))
    a.grid(True)

for i in range(nrows):
  for j in range(ncols):
    a = axarr[i, j]

    if i < 2:
      c = "BOX"
    else:
      c = "STAR"

    if i % 2 == 0:
      d = "independent"
    else:
      d = "dependent"

    a.set_title('%s sample (%s model)' % (c, d))


plt.show()
