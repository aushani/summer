import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

track_id = 5
frame_id = 36

fn = '/home/aushani/summer/cc/track_%03d_frame_%03d.csv' % (track_id, frame_id)

points = np.loadtxt(fn, delimiter=',')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2])

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
#ax.zlabel('z')

plt.show()
