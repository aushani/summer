import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path
import argparse

def show_map(ax, x, y, z, title, points=None, vmin=None, vmax=None):
  im = ax.pcolor(x, y, z, vmin=vmin, vmax=vmax)
  ax.scatter(0, 0, c='g', marker='x')
  plt.colorbar(im, ax=ax)

  if not points is None:
    ax.scatter(points[:, 0], points[:, 1], color='k', marker='.')

  ax.axis('equal')
  ax.axis((np.min(x[:]), np.max(x[:]), np.min(y[:]), np.max(y[:])))

  ax.set_title(title)

def show_detection(res, ax_score=None, ax_logodds=None, ax_prob=None, points=None, angle=0, do_non_max=True, name=''):
  unique_xs = np.unique(np.round(res[:, 0], decimals=2))
  unique_ys = np.unique(np.round(res[:, 1], decimals=2))
  unique_angles = np.unique(np.round(res[:, 2], decimals=2))

  angle_deg = unique_angles[angle] * 180 / np.pi

  grid_shape = (len(unique_xs), len(unique_ys), len(unique_angles))

  x = np.reshape(res[:, 0], grid_shape)
  y = np.reshape(res[:, 1], grid_shape)

  score = np.reshape(res[:, 3], grid_shape)
  logodds = np.reshape(res[:, 4], grid_shape)
  prob = np.reshape(res[:, 5], grid_shape)

  x_angle = x[:, :, angle]
  y_angle = y[:, :, angle]
  score_angle = score[:, :, angle]
  logodds_angle = logodds[:, :, angle]
  prob_angle = prob[:, :, angle]

  if not ax_score is None:
    show_map(ax_score,   x_angle, y_angle, score_angle   , '%s at %5.1f deg (score)' % (name, angle_deg), points=points)

  if not ax_logodds is None:
    show_map(ax_logodds, x_angle, y_angle, logodds_angle , '%s at %5.1f deg (log-odds)' % (name, angle_deg), points=points)

  if not ax_score is None:
    show_map(ax_prob,    x_angle, y_angle, prob_angle    , '%s at %5.1f deg (prob)' % (name, angle_deg), points=points, vmin=0, vmax=1)

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
  res = np.loadtxt('/home/aushani/summer/cc/result_%s_%03d.csv' % (class_name, experiment), delimiter=',')

  print 'Scores range from %5.3f to %5.3f'   % (np.min(res[:, 3]), np.max(res[:, 3]))
  print 'Log-odds range from %5.3f to %5.3f' % (np.min(res[:, 4]), np.max(res[:, 4]))
  print 'Probs range from %5.3f to %5.3f'    % (np.min(res[:, 5]), np.max(res[:, 5]))

  f_score, axarr_score = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
  f_logodds, axarr_logodds = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
  f_prob, axarr_prob = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
  #f_score, axarr_score = plt.subplots(nrows = 1, ncols = 1)
  #f_prob, axarr_prob = plt.subplots(nrows = 1, ncols = 1)

  #show_detection(res, ax_score=axarr_score, ax_prob=axarr_prob, points=points, angle = 0, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[0, 0], ax_logodds=axarr_logodds[0, 0], ax_prob=axarr_prob[0, 0], points=points, angle = 0, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[0, 1], ax_logodds=axarr_logodds[0, 1], ax_prob=axarr_prob[0, 1], points=points, angle = 1, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[0, 2], ax_logodds=axarr_logodds[0, 2], ax_prob=axarr_prob[0, 2], points=points, angle = 2, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[1, 0], ax_logodds=axarr_logodds[1, 0], ax_prob=axarr_prob[1, 0], points=points, angle = 3, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[1, 1], ax_logodds=axarr_logodds[1, 1], ax_prob=axarr_prob[1, 1], points=points, angle = 4, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[1, 2], ax_logodds=axarr_logodds[1, 2], ax_prob=axarr_prob[1, 2], points=points, angle = 5, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[2, 0], ax_logodds=axarr_logodds[2, 0], ax_prob=axarr_prob[2, 0], points=points, angle = 6, do_non_max = False, name=class_name)
  show_detection(res, ax_score=axarr_score[2, 1], ax_logodds=axarr_logodds[2, 1], ax_prob=axarr_prob[2, 1], points=points, angle = 7, do_non_max = False, name=class_name)

  f_score.delaxes(axarr_score[2, 2])
  f_logodds.delaxes(axarr_logodds[2, 2])
  f_prob.delaxes(axarr_prob[2, 2])

def show_model(model):
  f, ax = plt.subplots(nrows = 1, ncols = 1)

  unique_xs = np.unique(np.round(model[:, 0], decimals=2))
  unique_ys = np.unique(np.round(model[:, 1], decimals=2))
  grid_shape = (len(unique_xs), len(unique_ys))

  x = np.reshape(model[:, 0], grid_shape)
  y = np.reshape(model[:, 1], grid_shape)
  z = np.reshape(model[:, 2], grid_shape);

  #z = np.clip(z, 1e-10, None)
  #z = np.log(z)

  #sc = ax.scatter(x, y, c=z, marker='x', s=10)
  im = ax.pcolor(x, y, z);
  plt.colorbar(im, label='Likelihood')
  ax.axis('equal')
  #ax.axis((-5, 5, 0, 8))
  ax.grid(True)
  ax.set_title('Observation Model')

parser = argparse.ArgumentParser(description="Make some plots")
parser.add_argument('exp', metavar = 'exp', nargs='+', help='experiment number')
args = parser.parse_args()


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': '24',
         'axes.titlesize':'24',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)

experiment = int(args.exp[0])

points = np.loadtxt('/home/aushani/summer/cc/data_%03d.csv'         % (experiment), delimiter=',')
gt     = np.loadtxt('/home/aushani/summer/cc/ground_truth_%03d.csv' % (experiment), delimiter=',')

print 'Plotting...'

if True:
  f, axarr = plt.subplots(nrows = 1, ncols = 2)

  axarr[0].scatter(points[:, 0], points[:, 1], marker='.')
  axarr[0].scatter(0, 0, c='r', marker='x')
  axarr[0].axis((-20, 20, -20, 20))
  axarr[0].axis('equal')
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

if True:
  box  = np.loadtxt('/home/aushani/summer/cc/BOX.csv'                         , delimiter=',')
  star  = np.loadtxt('/home/aushani/summer/cc/STAR.csv'                         , delimiter=',')
  noobj  = np.loadtxt('/home/aushani/summer/cc/NOOBJ.csv'                         , delimiter=',')

  show_model(box)
  show_model(star)
  show_model(noobj)

print 'Ready to show, press Enter'
raw_input()
plt.show()
