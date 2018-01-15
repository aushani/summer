import numpy as np
import matplotlib.pyplot as plt
from model import *
from data import *

n_samples = 15
n_basis = n_samples
n_fit = 1000

# Ideal fit
x_ideal, y_ideal = generate_data(n_samples = n_fit, stddev = 0.0)

# Make data
x_samples, y_samples = generate_data(n_samples = n_samples, stddev = 0.2)
x_val, y_val = generate_data(n_samples = n_samples, stddev = 0.2)

# Get basis functions
x_m = np.linspace(0, 2*np.pi, num = n_samples)

# Build model
model = Model(x_samples, y_samples, x_m)

# Get fit
model.compute_least_squares_fit(l2_reg = 0)

# Evaluate
x_ls = x_ideal
y_ls = model.get_map_fit(x_ls)

l2_regs = np.logspace(-10, 2, 100)
train_errs = np.zeros(l2_regs.shape)
val_errs = np.zeros(l2_regs.shape)

for i, l2_reg in enumerate(l2_regs):
    model.compute_least_squares_fit(l2_reg = l2_reg)

    train_err = model.get_normalized_error(x_samples, y_samples)
    val_err = model.get_normalized_error(x_val, y_val)

    train_errs[i] = train_err
    val_errs[i] = val_err

# Plot
plt.subplot(3, 3, 1)
plt.scatter(x_samples, y_samples)
plt.scatter(x_val, y_val, c='g')
plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=4)
plt.grid(True)
plt.xlim((0, 2*np.pi))
plt.ylim((-1.5, 1.5))
plt.title('Samples and Ideal Fit')

plt.subplot(3, 3, 2)
plt.scatter(x_samples, y_samples)
plt.scatter(x_val, y_val, c='g')
plt.plot(x_ls, y_ls, 'r', linestyle='--', linewidth=2)
plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=1)
plt.grid(True)
plt.xlim((0, 2*np.pi))
plt.ylim((-1.5, 1.5))
plt.title('Least-Squares RBF Fit')

plt.subplot(3, 3, 3)
plt.semilogx(l2_regs, train_errs, 'b', label='Training')
plt.semilogx(l2_regs, val_errs, 'g', label='Validation')
plt.grid(True)
plt.title('Error')
plt.legend()

for sp in range(4, 10):
    idx = int((len(l2_regs) - 1) * (sp - 4.0) / 5.0)
    l2_reg = l2_regs[idx]
    model.compute_least_squares_fit(l2_reg = l2_reg)
    x_pls = x_ideal
    y_pls = model.get_map_fit(x_pls)

    plt.subplot(3, 3, sp)
    plt.scatter(x_samples, y_samples)
    plt.scatter(x_val, y_val, c='g')
    plt.plot(x_pls, y_pls, 'r', linestyle='--', linewidth=2)
    plt.plot(x_ideal, y_ideal, 'k', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.xlim((0, 2*np.pi))
    plt.ylim((-1.5, 1.5))
    plt.title('LS RBF Fit %5.1e' % (l2_reg))

plt.show()
