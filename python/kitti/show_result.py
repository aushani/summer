import numpy as np
import matplotlib.pyplot as plt
import pandas

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

classes = ["Car", "Pedestrian", "Cyclist", "NOOBJ"]
f, axarr = plt.subplots(nrows = 1, ncols = len(classes))

for i, classname in enumerate(classes):
  fn = '/home/aushani/summer/cc/result_%s.csv' % (classname)

  res = pandas.read_csv(fn, header=None).values
  print res.shape

  x = res[:, 0]
  y = res[:, 1]
  z = res[:, 2]
  t = res[:, 3]
  p = res[:, 6]

  unique_xs = np.unique(np.round(x, decimals=2))
  unique_ys = np.unique(np.round(y, decimals=2))
  unique_zs = np.unique(np.round(z, decimals=2))
  unique_ts = np.unique(np.round(t, decimals=2))

  shape = (len(unique_xs), len(unique_ys), len(unique_zs), len(unique_ts))

  print shape

  x = np.reshape(x, shape)
  y = np.reshape(y, shape)
  p = np.reshape(p, shape)
  print p.shape

  p = np.sum(p, axis=-1)
  print p.shape

  p = np.sum(p, axis=-1)
  print p.shape

  x = x[:, :, 0, 0]
  y = y[:, :, 0, 0]

  show_map(axarr[i], x, y, p, classname, cb = True)

plt.show()
