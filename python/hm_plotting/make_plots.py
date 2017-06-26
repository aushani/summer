import numpy as np
import matplotlib.pyplot as plt

print 'Loading...'
fit = np.loadtxt('grid.csv', delimiter=',')
points = np.loadtxt('points.csv', delimiter=',')

kernel = np.loadtxt('kernel.csv', delimiter=',')

print 'HM Range from %5.3f to %5.3f' % (np.min(fit[:, 2]), np.max(fit[:, 2]))
print 'Kernel Range from %f to %f' % (np.min(kernel[:, 2]), np.max(kernel[:, 2]))

print 'Plotting...'
f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(fit[:, 0], fit[:, 1], s=1, c=fit[:, 2], marker='.')
axarr[0].hold(True)
axarr[0].scatter(points[:, 0], points[:, 1], c='k', marker='.')
axarr[0].scatter(0, -2, c='r', marker='x')
axarr[0].axis('equal')
axarr[0].axis((np.min(fit[:, 0]), np.max(fit[:, 0]), np.min(fit[:, 1]), np.max(fit[:, 1])))

#axarr[1].scatter(kernel[:, 0], kernel[:, 1], s=4, c=kernel[:, 2], marker='.')
new_dim = int(round(kernel.shape[0]**0.5))
print new_dim
x = np.reshape(kernel[:, 0], (new_dim, new_dim))
y = np.reshape(kernel[:, 1], x.shape)
print x.shape
print y.shape
z = np.reshape(kernel[:, 2], x.shape)
print z.shape
axarr[1].pcolor(x, y, z)
axarr[1].axis((-2, 2, -2, 2))
axarr[1].axis('equal')


plt.show()
