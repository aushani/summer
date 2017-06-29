import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt('/home/aushani/summer/cc/points.csv', delimiter=',')
hm = np.loadtxt('/home/aushani/summer/cc/hilbert_map.csv', delimiter=',')

def show_grid(grid, ax, points=None):
  new_dim = int(round(grid.shape[0]**(0.5)))
  x = np.reshape(grid[:, 0], (new_dim, new_dim))
  y = np.reshape(grid[:, 1], (new_dim, new_dim))
  z = np.reshape(grid[:, 2], (new_dim, new_dim))

  im = ax.pcolor(x, y, z)
  plt.colorbar(im, ax=ax)

  #ax.scatter(grid[:, 0], grid[:, 1], c=grid[:, 2], marker='.', s=1)

  if points != None:
    ax.hold(True)
    ax.scatter(points[:, 0], points[:, 1], c='r', marker='.', s=10)
    ax.scatter(0, 0, c='r', marker='x')

  ax.axis('equal')
  ax.axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))

f, axarr = plt.subplots(nrows = 1, ncols = 2)
show_grid(hm, axarr[0])

axarr[1].scatter(points[:, 0], points[:, 1], c='r', marker='.', s=1)

plt.show()


