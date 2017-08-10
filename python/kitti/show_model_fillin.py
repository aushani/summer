import numpy as np
import matplotlib.pyplot as plt

#classes = ["Car", "Cyclist", "Pedestrian"]
classes = ["Car", "Cyclist"]

log_num = 1

f, axarr = plt.subplots(nrows = len(classes), ncols = 1, sharex=True, sharey=True)

for ax, classname in zip(axarr, classes):
  fn = '/home/aushani/summer/cc/%s_%02d.csv' % (classname, log_num)

  counts = np.loadtxt(fn, delimiter=",")

  sc = ax.scatter(counts[:, 0] * 180 / np.pi, counts[:, 1] * 180 / np.pi, c = counts[:, 2])
  ax.set_title(classname)
  ax.set_xlabel('Theta (relative to laser beam)')
  ax.set_xlim((-180, 180))
  ax.set_ylabel('Phi')
  ax.grid(True)

  plt.colorbar(sc, ax=ax, label='Histograms')

plt.show()
