import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os.path

from sklearn.metrics import precision_recall_curve

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': '24',
         'axes.titlesize':'24',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)

pr_scores  = np.loadtxt('/home/aushani/summer/cc/pr_scores.csv', delimiter=',')

precision, recall, thresholds = precision_recall_curve(pr_scores[:, 1], pr_scores[:, 0])

plt.plot(recall, precision)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0, 1.0])
plt.ylim([0, 1.05])
plt.show()
