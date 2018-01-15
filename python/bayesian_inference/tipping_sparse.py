import numpy as np
import matplotlib.pyplot as plt
import time
from model import *
from data import *

n_samples = 250
n_basis = 50
n_fit = 1000
stddev = 0.1

n_posterior_samples = 10

# Ideal fit
x_ideal, y_ideal = generate_data_2(n_samples = n_fit, stddev = 0.0)
x_fit = x_ideal

# Make data
x_samples, y_samples = generate_data_2(n_samples = n_samples, stddev = stddev)
x_val, y_val = generate_data_2(n_samples = n_samples, stddev = stddev)

# Get basis functions
#x_m = np.linspace(0, 2*np.pi, num = n_basis)
x_m = x_samples
n_basis = len(x_m)
x_m_idx = np.arange(0, n_basis)

# Build model
model = Model(x_samples, y_samples, x_m)

#model.compute_posterior(stddev, 1.0)
#print model.sigma
#
#model.compute_posterior(stddev, alphas, sparse=True)
#print model.sigma
#
#raw_input()

stddev0 = 1.0
alpha0 = np.ones((n_basis))

it_alpha = []
it_x_m = []

for it in range(1000):
    print 'Iteration %d, Have %d basis' % (it, len(x_m))
    params = model.get_params_mp(stddev0 = stddev0, alpha0 = alpha0, sparse=True, iterations=1)

    best_stddev = params[0]
    best_alphas = params[1]

    it_alpha.append(best_alphas)
    it_x_m.append(x_m)

    # Prune based on alpha
    basis_to_keep = best_alphas < 1e3
    best_alphas = best_alphas[basis_to_keep]
    x_m = x_m[basis_to_keep]
    x_m_idx = x_m_idx[basis_to_keep]
    model.update_x_m(x_m)

    stddev0 = best_stddev
    alpha0 = best_alphas

for a, b in zip(it_alpha, it_x_m):
    plt.semilogy(b, a, '.')

plt.grid(True)
plt.show()


print 'Have %d/%d basis functions left' % (len(x_m), n_basis)

model_pruned = Model(x_samples, y_samples, x_m)

# Now compute model with best parameters
model_pruned.compute_posterior(stddev = best_stddev, alpha = best_alphas, sparse=True)

posterior_samples = model_pruned.sample_fit(x_fit, num=n_posterior_samples)
print posterior_samples.shape

for i in range(n_posterior_samples):
    y_fit = posterior_samples[:, i]
    plt.plot(x_fit, y_fit, 'k', linewidth=0.1)

plt.scatter(x_samples, y_samples, c='b', s=5)
plt.scatter(x_samples[x_m_idx], y_samples[x_m_idx], marker='o', facecolors = 'none', edgecolors='r', s = 100)

# MAP estimate
y_fit = model_pruned.get_fit_mean(x_fit)
plt.plot(x_fit, y_fit, 'g', linestyle='--', linewidth=3)

# True fit
plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=2.0)

y_l1, y_u1 = model_pruned.get_bounds(x_ideal, stddev = 0.0, n = 1)
y_l3, y_u3 = model_pruned.get_bounds(x_ideal, stddev = 0.0, n = 3)
plt.plot(x_ideal, y_l1, 'b', linestyle='--', linewidth=1.0)
plt.plot(x_ideal, y_u1, 'b', linestyle='--', linewidth=1.0)
plt.plot(x_ideal, y_l3, 'r', linestyle='--', linewidth=1.0)
plt.plot(x_ideal, y_u3, 'r', linestyle='--', linewidth=1.0)

plt.grid(True)
plt.xlim((0, 2*np.pi))
plt.ylim((-2.5, 2.5))
plt.title('Model Samples')

plt.show()
