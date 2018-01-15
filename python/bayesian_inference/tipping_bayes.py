import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import *
from data import *

n_samples = 1000
n_basis = 15
n_fit = 1000
n_posterior_samples = 50

stddev = 0.2
alpha = 1

# Ideal fit
x_ideal, y_ideal = generate_data(n_samples = n_fit, stddev = 0.0)

x_fit = x_ideal

# Make data
x_full_samples, y_full_samples = generate_data(n_samples = n_samples, stddev = stddev)

# Get basis functions centers
x_m = np.linspace(0, 2*np.pi, num = n_basis)

# Plotting
num = [0, int(n_samples*0.5), int(n_samples*1.0), int(n_samples*1.0)]
step = [n_samples/15, n_samples/15, n_samples/15, 1]

plt_rows = 4
plt_cols = 4
for exp in range(plt_rows):
    print 'Generating samples'
    x_samples = x_full_samples[0:num[exp]:step[exp]]
    y_samples = y_full_samples[0:num[exp]:step[exp]]

    print 'Computing posterior'
    model = Model(x_samples, y_samples, x_m)
    model.compute_posterior(stddev = stddev, alpha=alpha)

    print 'Plotting samples'
    plt.subplot(plt_rows, plt_cols, plt_cols*exp + 1)
    if len(x_samples) < 30:
        plt.scatter(x_full_samples, y_full_samples, c='b', s=1)
        plt.scatter(x_samples, y_samples, c='r', marker='x')
    else:
        plt.scatter(x_full_samples, y_full_samples, c='b', s=1)
        plt.scatter(x_samples, y_samples, c='r', s=1)

    plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=2.0)
    plt.grid(True)
    plt.xlim((0, 2*np.pi))
    plt.ylim((-2.5, 2.5))
    plt.title('Samples')

    print 'Computing weight prior for w_10 and w_11'
    idx1 = 10
    idx2 = 11
    sigma = np.zeros((2, 2))
    sigma[0, 0] = model.sigma[idx1, idx1]
    sigma[1, 1] = model.sigma[idx2, idx2]
    sigma[0, 1] = model.sigma[idx1, idx2]
    sigma[1, 0] = model.sigma[idx2, idx1]

    eigenval, eigenvec = np.linalg.eig(sigma)
    ev = np.sqrt(eigenval)

    ax = plt.subplot(plt_rows, plt_cols, plt_cols*exp + 2, aspect='equal')
    for i in range(1, 4):
        ell = patches.Ellipse(xy=(model.mu[idx1], model.mu[idx2]),
                              width=ev[0]*i*2, height=ev[1]*i*2,
                              angle=np.rad2deg(np.arccos(eigenvec[0, 0])))
        ell.set_alpha(0.1)
        ax.add_artist(ell)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.title('Model Certainty')

    print 'Plotting posterior samples'
    plt.subplot(plt_rows, plt_cols, plt_cols*exp + 3)

    posterior_samples = model.sample_fit(x_fit, num=n_posterior_samples)
    print posterior_samples.shape

    for i in range(n_posterior_samples):
        y_fit = posterior_samples[:, i]
        plt.plot(x_fit, y_fit, 'k', linewidth=0.1)

    if len(x_samples) < 30:
        plt.scatter(x_samples, y_samples, c='b', s=5)

    # MAP estimate
    y_fit = model.get_fit_mean(x_fit)
    plt.plot(x_fit, y_fit, 'g', linestyle='--', linewidth=3)

    # True fit
    plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=2.0)

    y_l1, y_u1 = model.get_bounds(x_ideal, stddev = 0.0, n = 1)
    y_l3, y_u3 = model.get_bounds(x_ideal, stddev = 0.0, n = 3)
    plt.plot(x_ideal, y_l1, 'b', linestyle='--', linewidth=1.0)
    plt.plot(x_ideal, y_u1, 'b', linestyle='--', linewidth=1.0)
    plt.plot(x_ideal, y_l3, 'r', linestyle='--', linewidth=1.0)
    plt.plot(x_ideal, y_u3, 'r', linestyle='--', linewidth=1.0)

    plt.grid(True)
    plt.xlim((0, 2*np.pi))
    plt.ylim((-2.5, 2.5))
    plt.title('Model Samples')

    # Plot distributions
    y_l1, y_u1 = model.get_bounds(x_ideal, stddev = stddev, n = 1)
    y_l2, y_u2 = model.get_bounds(x_ideal, stddev = stddev, n = 2)
    y_l3, y_u3 = model.get_bounds(x_ideal, stddev = stddev, n = 3)

    plt.subplot(plt_rows, plt_cols, plt_cols*exp + 4)
    plt.plot(x_fit, y_fit, 'k', linewidth=2)
    plt.plot(x_ideal, y_l1, 'b', linewidth=0.5)
    plt.plot(x_ideal, y_u1, 'b', linewidth=0.5)
    plt.plot(x_ideal, y_l3, 'r', linewidth=0.5)
    plt.plot(x_ideal, y_u3, 'r', linewidth=0.5)
    plt.grid(True)
    plt.xlim((0, 2*np.pi))
    plt.ylim((-2.5, 2.5))
    plt.title('Target Distribution')

plt.show()
