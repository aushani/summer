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

    surface = np.zeros((len(thetas), len(phis)))
    for idx in range(counts.shape[1]):
      theta = counts[idx, 0]
      phi = counts[idx, 1]
      val = counts[idx, 2]

      surface_i = np.argmin(np.abs(theta - thetas))
      surface_j = np.argmin(np.abs(phi - phis))
      print surface_i, surface_j
      surface[surface_i, surface_j] = val

    im = ax.imshow(surface, origin='lower', extent=(np.min(thetas), np.max(thetas), np.min(phis), np.max(phis)),
          vmin=0, vmax=100)
    ax.set_title(classname)
    ax.set_xlabel('Theta (relative to laser beam)')
    ax.set_xlim((-180, 180))
    ax.set_ylabel('Phi')
    ax.grid(True)

    plt.colorbar(im, ax=ax, label='Fill-in %')

plt.show()
