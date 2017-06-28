import numpy as np
import matplotlib.pyplot as plt

print 'Loading...'
grid = np.loadtxt('/home/aushani/summer/cc/grid.csv', delimiter=',')
points = np.loadtxt('/home/aushani/summer/cc/points.csv', delimiter=',')
gt = np.loadtxt('/home/aushani/summer/cc/ground_truth.csv', delimiter=',')

kernel = np.loadtxt('/home/aushani/summer/cc/kernel.csv', delimiter=',')

print 'Grid shape', grid.shape
print 'Kernel shape', kernel.shape
print 'HM Range from %5.3f to %5.3f' % (np.min(grid[:, 2]), np.max(grid[:, 2]))
print 'Kernel Range from %f to %f' % (np.min(kernel[:, 2]), np.max(kernel[:, 2]))

print 'Plotting...'
f, axarr = plt.subplots(nrows=2, ncols=2)

axarr[0, 0].scatter(points[:, 0], points[:, 1], c='k', marker='.')
axarr[0, 0].scatter(0, 0, c='r', marker='x')
axarr[0, 0].axis((-15, 15, -15, 15))
axarr[0, 0].set_title('LIDAR')
axarr[0, 0].axis('equal')

axarr[1, 0].axis('equal')
axarr[1, 0].axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))
new_dim = int(round(grid.shape[0]**0.5))
axarr[1, 0].scatter(grid[:, 0], grid[:, 1], c=grid[:, 2], marker='.', s=1)
axarr[1, 0].hold(True)
axarr[1, 0].scatter(points[:, 0], points[:, 1], c='r', marker='.', s=10)
axarr[1, 0].scatter(0, 0, c='r', marker='x')
axarr[1, 0].axis('equal')
axarr[1, 0].axis((np.min(grid[:, 0]), np.max(grid[:, 0]), np.min(grid[:, 1]), np.max(grid[:, 1])))
axarr[1, 0].set_title('HM')
#plt.colorbar(p, ax=axarr[1])

axarr[0, 1].axis('equal')
axarr[0, 1].axis((-10, 10, -10, 10))
axarr[0, 1].scatter(gt[:, 0], gt[:, 1], c=gt[:, 2], marker='.')
axarr[0, 1].set_title('Ground Truth')
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
axarr[1, 1].set_title('Learned Kernel')
axarr[1, 1].grid(True)

plt.show()
