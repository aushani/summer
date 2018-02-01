from data_manager import *

batch_dir = '/home/aushani/data/batches/'

bm = BatchMaker('/home/aushani/data/auto_encoder_data_bin_buffer/', batch_size=100, n_test_samples=1000)
bm.generate_data_files(batch_dir)
