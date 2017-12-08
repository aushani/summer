import numpy as np
import matplotlib.pyplot as plt

x_samples = np.loadtxt('/home/aushani/summer/cc/data.csv', delimiter=',')
labels = np.loadtxt('/home/aushani/summer/cc/labels.csv', delimiter=',')

pred_labels = labels

def show_samples(x_samples, labels):
    pos = x_samples[labels < 0.5, :]
    neg = x_samples[labels > 0.5, :]

    plt.scatter(pos[:, 0], pos[:, 1], c='g', marker='x')
    plt.scatter(neg[:, 0], neg[:, 1], c='b', marker='o')

plt.subplot(2, 1, 1)
show_samples(x_samples, labels)
plt.grid(True)
plt.axis('equal')
plt.title('True')

plt.subplot(2, 1, 2)
show_samples(x_samples, pred_labels)
#plt.scatter(model.x_m[:, 0], model.x_m[:, 1], marker='o', facecolors='none', edgecolors='r', s=100)
plt.grid(True)
plt.axis('equal')
plt.title('Predicted')

plt.show()
