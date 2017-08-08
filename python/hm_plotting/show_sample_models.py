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


#classes = ['BOX', 'STAR', 'NOOBJ']
classes = ['BOX', 'STAR']

nrows = 2*len(classes)
ncols = 4

f, axarr = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True)

for j in range(ncols):
  idx = j

  for i, c in enumerate(classes):
    ind = np.loadtxt('/home/aushani/summer/cc/%s_sample_%d.csv' % (c, idx) , delimiter=',')
    dep = np.loadtxt('/home/aushani/summer/cc/%s_sample_dependent_%d.csv' % (c, idx) , delimiter=',')

    sc = axarr[2*i + 0, j].scatter(ind[:, 0], ind[:, 1], c = ind[:, 2])
    #plt.colorbar(sc, ax=axarr[2*i + 0, j])
    axarr[2*i + 0, j].axis("equal")
    axarr[2*i + 0, j].axis((-5, 5, 10, 20))
    axarr[2*i + 0, j].grid(True)
    axarr[2*i + 0, j].set_title('%s sample (1-gram)' % (c))

    cs = axarr[2*i + 1, j].scatter(dep[:, 0], dep[:, 1], c = dep[:, 2])
    #plt.colorbar(sc, ax=axarr[2*i + 1, j])
    axarr[2*i + 1, j].axis("equal")
    axarr[2*i + 1, j].axis((-5, 5, 10, 20))
    axarr[2*i + 1, j].grid(True)
    axarr[2*i + 1, j].set_title('%s sample (2-gram)' % (c))

box    = pandas.read_csv('/home/aushani/summer/cc/BOX.csv'   , header=None).values
star   = pandas.read_csv('/home/aushani/summer/cc/STAR.csv'  , header=None).values
noobj  = pandas.read_csv('/home/aushani/summer/cc/NOOBJ.csv' , header=None).values

show_model(box)
show_model(star)
show_model(noobj)


plt.show()
