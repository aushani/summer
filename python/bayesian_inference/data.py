import numpy as np

def generate_data(n_samples = 15, stddev = 0.2):
    x_samples = np.linspace(0, 2*np.pi, num = n_samples)
    y_samples = np.sin(x_samples) + np.random.normal(scale=stddev, size=n_samples)

    return x_samples, y_samples

def generate_data_2(n_samples = 15, stddev = 0.2):
    x_samples = np.linspace(0, 2*np.pi, num = n_samples)
    y_samples = np.sinc(x_samples - np.pi) + np.random.normal(scale=stddev, size=n_samples)

    return x_samples, y_samples

def generate_classification_samples(n_samples = 10, stddev = 0.2):
    sz = (n_samples, 2)
    #x_true = np.random.uniform(low=0, high = 1.0, size = sz)
    x_true = np.random.normal(scale = 0.25, size = sz)

    x = x_true[:, 0]
    y = x_true[:, 1]
    val = 0.2*np.sin(x*np.pi*2)

    labels = y > val
    labels = labels.astype(int)

    x_samples = x_true + np.random.normal(scale=stddev, size=sz)
    #x_samples[labels>0.5, 1] += 0.1

    return x_samples, labels

def generate_classification_samples_2(n_samples = 10, stddev = 0.2):
    sz = (n_samples, 2)
    #x_true = np.random.uniform(low=0, high = 1.0, size = sz)
    x_true = np.random.normal(scale = 0.25, size = sz)

    x = x_true[:, 0]
    y = x_true[:, 1]
    #val = 0.3*np.sin(x*np.pi*5) + 0.5
    val = np.square(x) + np.square(y)

    labels = val > 0.33**2
    labels = labels.astype(int)

    x_samples = x_true + np.random.normal(scale=stddev, size=sz)

    return x_samples, labels
