import numpy as np
import matplotlib.pyplot as plt
from data import *
from rvm import *

n_samples = 200
stddev = 0.05

def show_samples(x_samples, labels):
    pos = x_samples[labels < 0.5, :]
    neg = x_samples[labels > 0.5, :]

    plt.scatter(pos[:, 0], pos[:, 1], c='g', marker='x')
    plt.scatter(neg[:, 0], neg[:, 1], c='b', marker='o')

print 'Generating samples'
#x_samples, labels = generate_classification_samples(n_samples = n_samples, stddev = stddev)
x_samples, labels = generate_classification_samples_2(n_samples = n_samples, stddev = stddev)

print 'Initializing RVM'
model = RVM(x_samples, labels)

print 'Training'
model.update_params(iters=999)

print 'Predicting labels'
pred_labels = model.predict_labels(x_samples)
#pred_labels = labels

plt.subplot(2, 1, 1)
show_samples(x_samples, labels)
plt.grid(True)
plt.axis('equal')
plt.title('True')

plt.subplot(2, 1, 2)
show_samples(x_samples, pred_labels)
plt.scatter(model.x_m[:, 0], model.x_m[:, 1], marker='o', facecolors='none', edgecolors='r', s=100)
plt.grid(True)
plt.axis('equal')
plt.title('Predicted')

plt.show()
