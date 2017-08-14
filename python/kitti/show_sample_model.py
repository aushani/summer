import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser(description="Make some plots")
parser.add_argument('classname', metavar = 'classname', nargs='+', help='class id')
parser.add_argument('num', metavar = 'num', nargs='+', help='num')
args = parser.parse_args()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

classname = args.classname[0]
num = int(args.num[0])

fn = '/home/aushani/summer/cc/%s_%02d.csv' % (classname, num)

points = np.loadtxt(fn, delimiter=',')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2])

ax.set_xlim(-7.5, 7.5)
ax.set_ylim(5, 20)
ax.set_zlim(-5, 5)

plt.show()
