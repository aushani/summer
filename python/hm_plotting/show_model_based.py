import numpy as np
import matplotlib.pyplot as plt
import os.path

def show_model(model, ax_o, ax_f):
  new_dim = int(round(model.shape[0]**(0.5)))
  x = np.reshape(model[:, 0], (new_dim, new_dim))
  y = np.reshape(model[:, 1], (new_dim, new_dim))

  for ax, dim in zip([ax_o, ax_f], [2, 3]):
    p_z = np.reshape(model[:, dim], (new_dim, new_dim))

    im = ax.pcolor(x, y, p_z)
    plt.colorbar(im, ax=ax, label='Prob')

    ax.axis('equal')
    ax.axis((np.min(model[:, 0]), np.max(model[:, 0]), np.min(model[:, 1]), np.max(model[:, 1])))

def show_detection(res, ax_binary, ax_score):
  new_dim = int(round(res.shape[0]**(0.5)))
  x = np.reshape(res[:, 0], (new_dim, new_dim))
  y = np.reshape(res[:, 1], (new_dim, new_dim))

  binary = np.reshape(res[:, 2], (new_dim, new_dim))
  #score = np.reshape(res[:, 3], (new_dim, new_dim))
  score = binary

  im = ax_binary.pcolor(x, y, binary, cmap=plt.cm.binary)
  ax_binary.scatter(0, 0, c='r', marker='x')
  plt.colorbar(im, ax=ax_binary, label='Binary')

  im = ax_score.pcolor(x, y, score)
  ax_score.scatter(0, 0, c='r', marker='x')
  plt.colorbar(im, ax=ax_score, label='Prob')

  ax_binary.axis('equal')
  ax_binary.axis((np.min(res[:, 0]), np.max(res[:, 0]), np.min(res[:, 1]), np.max(res[:, 1])))

  ax_score.axis('equal')
  ax_score.axis((np.min(res[:, 0]), np.max(res[:, 0]), np.min(res[:, 1]), np.max(res[:, 1])))

  ax_binary.set_title('Detection (Binary)')
  ax_score.set_title('Detection (score)')

f, axarr = plt.subplots(nrows = 3, ncols = 2)

points = np.loadtxt('/home/aushani/summer/cc/data.csv', delimiter=',')
gt = np.loadtxt('/home/aushani/summer/cc/ground_truth.csv', delimiter=',')
res = np.loadtxt('/home/aushani/summer/cc/result.csv', delimiter=',')

print 'Detections scores range from %f to %f' % (np.min(res[:, 2]), np.max(res[:, 2]))
#print 'Detections scores range from %f to %f' % (np.min(res[:, 3]), np.max(res[:, 3]))

print 'Plotting...'

print  points.shape

axarr[0, 0].scatter(points[:, 0], points[:, 1], marker='.')
axarr[0, 0].scatter(0, 0, c='r', marker='x')
axarr[0, 0].axis('equal')
axarr[0, 0].axis((-10, 10, -10, 10))
axarr[0, 0].set_title('Data')

axarr[0, 1].scatter(gt[:, 0], gt[:, 1], c=gt[:, 2], marker='.')
axarr[0, 1].scatter(0, 0, c='r', marker='x')
axarr[0, 1].axis((-10, 10, -10, 10))
axarr[0, 1].axis('equal')
axarr[0, 1].set_title('Ground Truth')

show_detection(res, axarr[1, 0], axarr[1, 1])

#show_model(model, axarr[2, 0], axarr[2, 1])
#axarr[2, 0].set_title('Object Observation Model (Occu)')
#axarr[2, 1].set_title('Object Observation Model (Free)')

#show_model(noobj, axarr[3, 0], axarr[3, 1])
#axarr[3, 0].set_title('No Object Observation Model (Occu)')
#axarr[3, 1].set_title('No Object Observation Model (Free)')

#axarr[4, 0].scatter(occu[:, 0], occu[:, 1], c=occu[:, 2], s=1, marker='.')
#axarr[4, 0].set_title('Occupancy Probability')

plt.show()
