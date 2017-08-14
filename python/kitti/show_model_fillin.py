import numpy as np
import matplotlib.pyplot as plt
import os.path

#classes = ["Car", "Cyclist", "Pedestrian", "Tram", "Truck"]
classes = ["Car", "Cyclist", "Pedestrian", "NOOBJ"]
prefixes = ["raw", "blurred"]

f1, ax1 = plt.subplots(nrows = len(classes), ncols = len(prefixes), sharex=True, sharey=True)
f2, ax2 = plt.subplots(nrows = len(classes), ncols = len(prefixes), sharex=True, sharey=True)

for i, classname in enumerate(classes):
  for j, prefix in enumerate(prefixes):
    fn = '/home/aushani/summer/cc/%s_%s.csv' % (prefix, classname)

    if not os.path.isfile(fn):
      continue

    counts = np.loadtxt(fn, delimiter=",")

    thetas = np.unique(np.round(counts[:, 0] * 180 / np.pi, decimals = 2))
    phis   = np.unique(np.round(counts[:, 1] * 180 / np.pi, decimals = 2))

    x1 = np.min(thetas[:])
    x2 = np.max(thetas[:])

    y1 = np.min(phis[:])
    y2 = np.max(phis[:])

    shape = (len(thetas), len(phis))

    idx_zero = counts[:, 2] == 0
    #counts[idx_zero, 2] = np.nan
    fillin = np.reshape(100*counts[:, 2], shape)
    median_weight = np.log(np.reshape(counts[:, 3], shape))

    im1 = ax1[i, j].imshow(np.transpose(fillin), origin='lower', extent=(x1, x2, y1, y2), vmin=0, vmax=100)
    ax1[i, j].set_title(classname)
    ax1[i, j].set_xlim((-180, 180))
    ax1[i, j].set_ylim((-20, 5))
    ax1[i, j].grid(True)

    im2 = ax2[i, j].imshow(np.transpose(median_weight), origin='lower', extent=(x1, x2, y1, y2))
    ax2[i, j].set_title(classname)
    ax2[i, j].set_xlim((-180, 180))
    ax2[i, j].set_ylim((-20, 5))
    ax2[i, j].grid(True)

    if i==len(classes)-1:
      ax1[i, j].set_xlabel('Theta (relative to laser beam)')
      ax2[i, j].set_xlabel('Theta (relative to laser beam)')

    if j==0:
      ax1[i, j].set_ylabel('Phi')
      ax2[i, j].set_ylabel('Phi')

    plt.colorbar(im1, ax=ax1[i, j], label='Fill-in %')
    plt.colorbar(im2, ax=ax2[i, j], label='Log Median Weight')

plt.show()
