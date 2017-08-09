import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser(description="Make some plots")
parser.add_argument('track_id', metavar = 'track_id', nargs='+', help='track id')
parser.add_argument('frame_id', metavar = 'frame_id', nargs='+', help='frame id')
args = parser.parse_args()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

track_id = int(args.track_id[0])
frame_id = int(args.frame_id[0])

fn = '/home/aushani/summer/cc/track_%03d_frame_%03d_synth.csv' % (track_id, frame_id)

points = np.loadtxt(fn, delimiter=',')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2])

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

plt.show()
