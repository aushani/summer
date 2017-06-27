import numpy as np
import matplotlib.pyplot as plt

print 'Loading...'
grid = np.loadtxt('/home/aushani/summer/cc/grid.csv', delimiter=',')
points = np.loadtxt('/home/aushani/summer/cc/points.csv', delimiter=',')
scoring = np.loadtxt('/home/aushani/summer/cc/scoring.csv', delimiter=',')

kernel = np.loadtxt('/home/aushani/summer/cc/kernel.csv', delimiter=',')

print 'Grid shape', grid.shape
print 'Kernel shape', kernel.shape
print 'HM Range from %5.3f to %5.3f' % (np.min(grid[:, 2]), np.max(grid[:, 2]))
print 'Kernel Range from %f to %f' % (np.min(kernel[:, 2]), np.max(kernel[:, 2]))

print 'Plotting...'
f, axarr = plt.subplots(nrows=2, ncols=2)

axarr[0, 0].scatter(points[:, 0], points[:, 1], c='k', marker='.')
axarr[0, 0].scatter(0, -2, c='r', marker='x')
axarr[0, 0].set_title('LIDAR')

axarr[1, 0].axis('equal')
axarr[1, 0].axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))
new_dim = int(round(grid.shape[0]**0.5))
x = np.reshape(grid[:, 0], (new_dim, new_dim))
y = np.reshape(grid[:, 1], x.shape)
z = np.reshape(grid[:, 2], x.shape)
im = axarr[1, 0].pcolor(x, y, z)
#axarr[1, 0].scatter(grid[:, 0], grid[:, 1], c=grid[:, 2], marker='.')
plt.colorbar(im, ax=axarr[1, 0])
axarr[1, 0].hold(True)
#p = axarr[1, 0].scatter(points[:, 0], points[:, 1], c='k', marker='.', s=1)
axarr[1, 0].axis('equal')
axarr[1, 0].axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))
axarr[1, 0].set_title('HM')
#plt.colorbar(p, ax=axarr[1])

axarr[0, 1].axis('equal')
axarr[0, 1].axis((-10, 10, -10, 10))
axarr[0, 1].scatter(scoring[:, 0], scoring[:, 1], c=scoring[:, 2], marker='.')
axarr[0, 1].set_title('Scoring')
#plt.colorbar(p, ax=axarr[1])

#axarr[1].scatter(kernel[:, 0], kernel[:, 1], s=4, c=kernel[:, 2], marker='.')
new_dim = int(round(kernel.shape[0]**0.5))
x = np.reshape(kernel[:, 0], (new_dim, new_dim))
y = np.reshape(kernel[:, 1], x.shape)
z = np.reshape(kernel[:, 2], x.shape)
im = axarr[1, 1].pcolor(x, y, z)
plt.colorbar(im, ax=axarr[1, 1])
axarr[1, 1].axis((-2, 2, -2, 2))
axarr[1, 1].axis('equal')
axarr[1, 1].set_title('Kernel')
axarr[1, 1].grid(True)

plt.show()
