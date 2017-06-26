import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('points.csv', delimiter=',')
plt.scatter(x[:, 0], x[:, 1])
plt.hold(True)
plt.scatter(0, -2)
plt.axis((-10, 10, -10, 10))
plt.grid(True)
plt.show()

