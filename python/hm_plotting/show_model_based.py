import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse
import pandas

def show_map(ax, x, y, z, title, points=None, vmin=None, vmax=None, cb=False):
  x1 = np.min(x[:])
  x2 = np.max(x[:])

  y1 = np.min(y[:])
  y2 = np.max(y[:])

  #im = ax.pcolor(x, y, z, vmin=vmin, vmax=vmax)
  im = ax.imshow(np.transpose(z), origin='lower', extent=(x1, x2, y1, y2), vmin=vmin, vmax=vmax)
  ax.scatter(0, 0, c='g', marker='x')

  if cb:
    plt.colorbar(im, ax=ax)

  if not points is None:
    ax.scatter(points[:, 0], points[:, 1], color='k', marker='.')

  #ax.axis('equal')
  ax.axis((x1, x2, y1, y2))

  ax.set_title(title)

def show_detection(res, ax_score=None, ax_logodds=None, ax_prob=None, points=None, angle=0, do_non_max=True, name='', cb=False):
  unique_xs = np.unique(np.round(res[:, 0], decimals=2))
  unique_ys = np.unique(np.round(res[:, 1], decimals=2))
  unique_angles = np.unique(np.round(res[:, 2], decimals=2))

  grid_shape = (len(unique_xs), len(unique_ys), len(unique_angles))

  x = np.reshape(res[:, 0], grid_shape)
  y = np.reshape(res[:, 1], grid_shape)

  score = np.reshape(res[:, 3], grid_shape)
  logodds = np.reshape(res[:, 4], grid_shape)
  prob = np.reshape(res[:, 5], grid_shape)

  if angle>=0:
    x_angle = x[:, :, angle]
    y_angle = y[:, :, angle]
    score_angle = score[:, :, angle]
    logodds_angle = logodds[:, :, angle]
    prob_angle = prob[:, :, angle]

    angle_deg = unique_angles[angle] * 180 / np.pi

    if not ax_score is None:
      show_map(ax_score,   x_angle, y_angle, score_angle   , '%s at %5.0f deg (score)' % (name, angle_deg),
          points=points, cb=cb)

    if not ax_logodds is None:
      show_map(ax_logodds, x_angle, y_angle, logodds_angle , '%s at %5.0f deg (log-odds)' % (name, angle_deg), points=points, vmin=-40, vmax=40, cb=cb)

    if not ax_prob is None:
      show_map(ax_prob,    x_angle, y_angle, prob_angle    , '%s at %5.0f deg (prob)' % (name, angle_deg), points=points, vmin=0, vmax=1, cb=cb)

  else:
    x = x[:, :, 0]
    y = y[:, :, 0]
    prob = np.sum(prob, axis=-1)

    if not ax_prob is None:
      show_map(ax_prob, x, y, prob, '%s (prob)' % (name), points=points, vmin=0, vmax=1, cb=cb)


  # Non maximal supression
  if do_non_max:
    for i in range(grid_shape[0]):
      for j in range(grid_shape[1]):
        val = score[i, j, angle]
        if val < 100:
          continue

        window_size = 5

        i0 = i - window_size
        if i0 < 0:
          i0 = 0
        j0 = j - window_size
        if j0 < 0:
          j0 = 0
        i_max = i + window_size
        if i_max > grid_shape[0]:
          i_max = grid_shape[0]
        j_max = j + window_size
        if j_max > grid_shape[1]:
          j_max = grid_shape[1]

        is_max = True

        for im in range(i0, i_max):
          for jm in range(j0, j_max):
            for k in range(grid_shape[2]):
              if score[im, jm, k] > val:
                is_max = False
                break

            if is_max is False:
              break

          if is_max is False:
            break

        if is_max:
          print 'max at %5.3f, %5.3f, %f = %f' % (x[i, j, angle], y[i, j, angle], unique_angles[angle] * 180/np.pi, val)

          if ax_score:
            ax_score.scatter(x[i, j, angle], y[i, j, angle], c='r', marker='x', s=50)
          if ax_prob:
            ax_prob.scatter(x[i, j, angle], y[i, j, angle], c='r', marker='x', s=50)

def show_detection_layer(class_name, points):
  print 'Plotting', class_name
  filename = '/home/aushani/summer/cc/result_%s_%03d.csv' % (class_name, experiment)
  df = pandas.read_csv(filename, header=None)
  res = df.values

  print 'Scores range from %5.3f to %5.3f'   % (np.min(res[:, 3]), np.max(res[:, 3]))
  print 'Log-odds range from %5.3f to %5.3f' % (np.min(res[:, 4]), np.max(res[:, 4]))
  print 'Probs range from %5.3f to %5.3f'    % (np.min(res[:, 5]), np.max(res[:, 5]))

  if class_name is "NOOBJ":
    nrows = 1
    ncols = 1
  else:
    nrows = 5
    ncols = 5

  #f_score, axarr_score = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, sharey=True)
  f_logodds, axarr_logodds = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, sharey=True)
  f_prob, axarr_prob = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, sharey=True)


  if class_name is "NOOBJ":
    show_detection(res, ax_score=None, ax_logodds=axarr_logodds, ax_prob=axarr_prob,
        points=points, angle = 0, do_non_max = False, name=class_name, cb=True)
  else:
    for i in range(nrows):
      for j in range(ncols):
        angle = i*ncols + j
        show_detection(res, ax_score=None, ax_logodds=axarr_logodds[i, j], ax_prob=axarr_prob[i, j],
            points=points, angle = angle, do_non_max = False, name=class_name, cb=True)

    f_sumprob, axarr_sumprob = plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey=True)
    show_detection(res, ax_prob=axarr_sumprob, points=points, angle = -1, do_non_max = False, name=class_name, cb=True)

parser = argparse.ArgumentParser(description="Make some plots")
parser.add_argument('exp', metavar = 'exp', nargs='+', help='experiment number')
args = parser.parse_args()


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': '16',
         'axes.titlesize':'16',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)

experiment = int(args.exp[0])

print 'Plotting...'

points = np.loadtxt('/home/aushani/summer/cc/data_%03d.csv'         % (experiment), delimiter=',')

if True:
  gt     = np.loadtxt('/home/aushani/summer/cc/ground_truth_%03d.csv' % (experiment), delimiter=',')

  f, axarr = plt.subplots(nrows = 1, ncols = 2)

  axarr[0].scatter(points[:, 0], points[:, 1], marker='.')
  axarr[0].scatter(0, 0, c='r', marker='x')
  axarr[0].axis('equal')
  axarr[0].axis((-20, 20, -20, 20))
  axarr[0].grid(True)
  axarr[0].set_title('LIDAR Data')

  axarr[1].scatter(gt[:, 0], gt[:, 1], c=gt[:, 2], marker='.')
  axarr[1].scatter(0, 0, c='r', marker='x')
  axarr[1].axis((-20, 20, -20, 20))
  axarr[1].axis('equal')
  axarr[1].grid(True)
  axarr[1].set_title('Ground Truth')

if True:
  show_detection_layer("BOX", points)
  show_detection_layer("STAR", points)
  show_detection_layer("NOOBJ", points)

print 'Ready to show, press Enter'
raw_input()
print 'Showing...'
plt.show()
