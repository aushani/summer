import numpy as np
import matplotlib.pyplot as plt
from model import *
from data import *

n_samples = 100
n_basis = 15
n_fit = 1000
stddev = 0.2

n_posterior_samples = 10

# Ideal fit
x_ideal, y_ideal = generate_data(n_samples = n_fit, stddev = 0.0)
x_fit = x_ideal

# Make data
x_samples, y_samples = generate_data(n_samples = n_samples, stddev = stddev)
x_val, y_val = generate_data(n_samples = n_samples, stddev = stddev)

# Get basis functions
x_m = np.linspace(0, 2*np.pi, num = n_basis)

# Build model
model = Model(x_samples, y_samples, x_m)

params = model.get_params_mp()

best_stddev = params[0]
best_alpha = params[1]

print 'Best Std. Dev. = %5.3f, Best Alpha = %5.3f' % (best_stddev, best_alpha)

# Now compute model with best parameters
model.compute_posterior(stddev = best_stddev, alpha = best_alpha)

posterior_samples = model.sample_fit(x_fit, num=n_posterior_samples)
print posterior_samples.shape

for i in range(n_posterior_samples):
    y_fit = posterior_samples[:, i]
    plt.plot(x_fit, y_fit, 'k', linewidth=0.1)

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

plt.show()
