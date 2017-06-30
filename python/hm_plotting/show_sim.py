import numpy as np
import matplotlib.pyplot as plt
import os.path

def show_kernel(kernel, ax):
  new_dim = int(round(kernel.shape[0]**0.5))
  x = np.reshape(kernel[:, 0], (new_dim, new_dim))
  y = np.reshape(kernel[:, 1], x.shape)
  z = np.reshape(kernel[:, 2], x.shape)
  im = ax.pcolor(x, y, z)
  plt.colorbar(im, ax=ax)
  ax.axis((-2, 2, -2, 2))
  ax.axis('equal')
  ax.grid(True)

def show_grid(grid, ax, points=None):
  new_dim = int(round(grid.shape[0]**(0.5)))
  if new_dim * new_dim == grid.shape[0]:
    x = np.reshape(grid[:, 0], (new_dim, new_dim))
    y = np.reshape(grid[:, 1], (new_dim, new_dim))
    z = np.reshape(grid[:, 2], (new_dim, new_dim))

    im = ax.pcolor(x, y, z)
    plt.colorbar(im, ax=ax)
  else:
    ax.scatter(grid[:, 0], grid[:, 1], c=grid[:, 2], marker='.', s=1)

  if points != None:
    ax.hold(True)
    ax.scatter(points[:, 0], points[:, 1], c='r', marker='.', s=10)
    ax.scatter(0, 0, c='r', marker='x')

  ax.axis('equal')
  ax.axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))

examples = 3

f, axarr = plt.subplots(nrows = 3, ncols = (examples+1))

print "kernel..."

for i, epoch in enumerate([1, 2]):
  fn = '/home/aushani/summer/cc/kernel_%04d.csv' % (epoch)
  if os.path.isfile(fn):
    kernel = np.loadtxt(fn, delimiter=',')
    show_kernel(kernel, axarr[i, 0])
    axarr[i, 0].set_title('Kernel after %d epochs' % (epoch))

kernel = np.loadtxt('/home/aushani/summer/cc/kernel.csv', delimiter=',')
show_kernel(kernel, axarr[2, 0])
axarr[2, 0].set_title('Learned Kernel')

for ex in range(examples):

  print 'Loading %d / %d ...' % (ex, examples)

  points = np.loadtxt('/home/aushani/summer/cc/points_%02d.csv' % (ex), delimiter=',')
  gt = np.loadtxt('/home/aushani/summer/cc/ground_truth_%02d.csv' % (ex), delimiter=',')
  hm = np.loadtxt('/home/aushani/summer/cc/hilbert_map_%02d.csv' % (ex), delimiter=',')

  #print 'Grid shape', grid.shape
  #print 'Kernel shape', kernel.shape
  #print 'HM Range from %5.3f to %5.3f' % (np.min(grid[:, 2]), np.max(grid[:, 2]))
  #print 'Kernel Range from %f to %f' % (np.min(kernel[:, 2]), np.max(kernel[:, 2]))

  print 'Plotting...'

  axarr[0, ex+1].scatter(points[:, 0], points[:, 1], c='k', marker='.')
  axarr[0, ex+1].scatter(0, 0, c='r', marker='x')
  axarr[0, ex+1].axis('equal')
  axarr[0, ex+1].axis((-10, 10, -10, 10))
  axarr[0, ex+1].set_title('LIDAR')

  show_grid(hm, axarr[1, ex+1])
  axarr[1, ex+1].set_title('HM')

  show_grid(gt, axarr[2, ex+1])
  axarr[2, ex+1].set_title('Ground Truth')

plt.show()
