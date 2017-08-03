import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse
import pandas

def show_model(model):
  f, ax = plt.subplots(nrows = 1, ncols = 1)

  unique_xs = np.unique(np.round(model[:, 0], decimals=2))
  unique_ys = np.unique(np.round(model[:, 1], decimals=2))
  grid_shape = (len(unique_xs), len(unique_ys))

  z = np.reshape(model[:, 2], grid_shape);
  z = np.log(z)

  x1 = np.min(model[:, 0])
  x2 = np.max(model[:, 0])

  y1 = np.min(model[:, 1])
  y2 = np.max(model[:, 1])

  im = ax.imshow(np.transpose(z), origin='lower', extent=(x1, x2, y1, y2))
  plt.colorbar(im, label='Log Prob')
  ax.axis('equal')
  #ax.axis((-5, 5, 0, 8))
  ax.grid(True)
  ax.set_title('Observation Model')


nrows = 6
ncols = 4

f, axarr = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)

for j in range(ncols):
  idx = j

  box_ind    = np.loadtxt('/home/aushani/summer/cc/BOX_sample_%d.csv' % idx , delimiter=',')
  box_dep    = np.loadtxt('/home/aushani/summer/cc/BOX_sample_dependent_%d.csv' % idx , delimiter=',')

  star_ind    = np.loadtxt('/home/aushani/summer/cc/STAR_sample_%d.csv' % idx , delimiter=',')
  star_dep    = np.loadtxt('/home/aushani/summer/cc/STAR_sample_dependent_%d.csv' % idx , delimiter=',')

  noobj_ind  = np.loadtxt('/home/aushani/summer/cc/NOOBJ_sample_%d.csv' % idx , delimiter=',')
  noobj_dep  = np.loadtxt('/home/aushani/summer/cc/NOOBJ_sample_dependent_%d.csv' % idx , delimiter=',')

  data = [box_ind, box_dep, star_ind, star_dep, noobj_ind, noobj_dep]

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

box    = pandas.read_csv('/home/aushani/summer/cc/BOX.csv'   , header=None).values
star   = pandas.read_csv('/home/aushani/summer/cc/STAR.csv'  , header=None).values
noobj  = pandas.read_csv('/home/aushani/summer/cc/NOOBJ.csv' , header=None).values

show_model(box)
show_model(star)
show_model(noobj)


plt.show()
