import numpy as np
import matplotlib.pyplot as plt
import os.path

classes = ["Car", "Cyclist", "Pedestrian", "Tram", "Truck"]
prefixes = ["raw", "blurred"]

f, axarr = plt.subplots(nrows = len(classes), ncols = len(prefixes), sharex=True, sharey=True)

for i, classname in enumerate(classes):
  for j, prefix in enumerate(prefixes):
    fn = '/home/aushani/summer/cc/%s_%s.csv' % (prefix, classname)

    if not os.path.isfile(fn):
      continue

    ax = axarr[i, j]

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

    im = ax.imshow(np.transpose(fillin), origin='lower',
        extent=(x1, x2, y1, y2), vmin=0, vmax=100)
    ax.set_title(classname)
    ax.set_xlim((-180, 180))
    ax.set_ylim((-20, 5))
    ax.grid(True)

    if i==len(classes)-1:
      ax.set_xlabel('Theta (relative to laser beam)')

    if j==0:
      ax.set_ylabel('Phi')

    plt.colorbar(im, ax=ax, label='Fill-in %')

plt.show()
