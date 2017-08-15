import numpy as np
import matplotlib.pyplot as plt
import pandas
import os.path

def show_map(ax, x, y, z, title, vmin=None, vmax=None, cb=False):
  x1 = np.min(x[:])
  x2 = np.max(x[:])

  y1 = np.min(y[:])
  y2 = np.max(y[:])

  #im = ax.pcolor(x, y, z, vmin=vmin, vmax=vmax)
  im = ax.imshow(np.transpose(z), origin='lower', extent=(x1, x2, y1, y2), vmin=vmin, vmax=vmax)
  ax.scatter(0, 0, c='g', marker='x')

  if cb:
    plt.colorbar(im, ax=ax)

  #ax.axis('equal')
  ax.axis((x1, x2, y1, y2))

  ax.set_title(title)

classes = ["Car", "Pedestrian", "Cyclist", "Tram", "Van", "NOOBJ"]
classes_exist = []

for i, classname in enumerate(classes):
  fn = '/home/aushani/summer/cc/result_%s.csv' % (classname)

  if not os.path.isfile(fn):
    continue

  classes_exist.append(classname)

classes = classes_exist
f, axarr = plt.subplots(nrows = 1, ncols = len(classes), sharex = True, sharey = True)

gt = pandas.read_csv('/home/aushani/summer/cc/ground_truth.csv', header=None).values

for i, classname in enumerate(classes):
  fn = '/home/aushani/summer/cc/result_%s.csv' % (classname)

  if not os.path.isfile(fn):
    continue

  res = pandas.read_csv(fn, header=None).values

  x = res[:, 0]
  y = res[:, 1]
  t = res[:, 2]
  lo = res[:, 4]
  p = res[:, 5]

  unique_xs = np.unique(np.round(x, decimals=2))
  unique_ys = np.unique(np.round(y, decimals=2))
  unique_ts = np.unique(np.round(t, decimals=2))

  shape = (len(unique_xs), len(unique_ys), len(unique_ts))

  x = np.reshape(x, shape)
  y = np.reshape(y, shape)
  p = np.reshape(p, shape)

  p = np.sum(p, axis=-1)
  print p.shape

  x = x[:, :, 0]
  y = y[:, :, 0]

  show_map(axarr[i], x, y, p, classname, vmin = 0, vmax = 1, cb = True)
  for x in gt:
    if x[3] != classname:
      continue

    axarr[i].scatter(x[0], x[1], c='r', marker='x')

plt.show()
