from auto_encoder import *
from data_manager import *
import matplotlib.pyplot as plt
import sys

batch_dir = '/home/aushani/data/batches_ae_kitti'

plt.switch_backend('agg')

ae = AutoEncoder()

if len(sys.argv) > 1:
    last_iter = int(sys.argv[1])
    iteration_start = last_iter + 1

    print 'Resume from iteration %d' % (last_iter)

    dm = DataManager(batch_dir, start_at = iteration_start)
    ae.restore('model_%08d.ckpt' % (last_iter))
    ae.train(dm, iteration=iteration_start)

else:
    print 'Starting afresh'

    dm = DataManager(batch_dir)
    ae.train(dm)
