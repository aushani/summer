from data_manager import *

batch_dir = '/home/aushani/data/batches_ae_kitti/'

bm = BatchMaker('/home/aushani/data/ae_kitti/', batch_size=100, n_test_samples=10000)
bm.generate_data_files(batch_dir)
