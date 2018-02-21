import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from auto_encoder import *
import time

def save_occ_grid(sample, fn):
    dim_x, dim_y, dim_z = sample.shape

    f = open(fn, 'w')
    f.write(np.array(sample.shape).tobytes())
    f.write(sample.tobytes())
    f.close()


# Load
print 'Loading Data'
batch_dir = '/home/aushani/data/batches_ae_kitti'
data_manager = DataManager(batch_dir)

print 'Loading AE'
ae = AutoEncoder()
ae.restore("dkblast/model_32660000.ckpt")

test_samples, test_labels = data_manager.test_samples, data_manager.test_labels_oh

# Car samples
idx_car = test_labels[:, 1] == 1
test_samples_car = test_samples[idx_car, :, :, :]
test_labels_car = test_labels[idx_car, :]

# Cyclist samples
idx_cyclist = test_labels[:, 2] == 1
test_samples_cyclist = test_samples[idx_cyclist, :, :, :]
test_labels_cyclist = test_labels[idx_cyclist, :]

print 'Reconstructing...'
reconstructed, pred_label, loss = ae.reconstruct_and_classify(test_samples_car)

save_occ_grid(reconstructed[0, :, :, :], "first.ae")
