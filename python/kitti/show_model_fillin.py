import numpy as np
import matplotlib.pyplot as plt
import os.path

classes = ["Car", "Cyclist", "Pedestrian"]
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

    print thetas
    print phis

    x1 = np.min(thetas[:])
    x2 = np.max(thetas[:])

    y1 = np.min(phis[:])
    y2 = np.max(phis[:])

    shape = (len(thetas), len(phis))
    print shape, len(thetas)*len(phis)
    print counts.shape

    fillin = np.reshape(counts[:, 2], shape)

    im = ax.imshow(fillin, origin='lower', extent=(x1, x2, y1, y2), vmin=0, vmax=100)
    ax.set_title(classname)
    ax.set_xlabel('Theta (relative to laser beam)')
    ax.set_xlim((-180, 180))
    ax.set_ylabel('Phi')
    ax.grid(True)

    plt.colorbar(im, ax=ax, label='Fill-in %')

plt.show()
