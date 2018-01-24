import numpy as np
import struct

class Grid:

    def __init__(self, path, dim_grid):
        fp = open(path, 'r')

        # We have a dense array of voxels
        num_voxels = dim_grid*dim_grid
        grid_bin = struct.unpack('f'*num_voxels, fp.read(4*num_voxels))

        # Done with file
        fp.close()

        grid = np.asarray(grid_bin)
        self.grid = np.reshape(grid, [dim_grid, dim_grid])

    def get_sample(self, xc, yc, dim_input=16*3*2-1):
        x0 = xc - dim_input/2
        x1 = x0 + dim_input

        y0 = yc - dim_input/2
        y1 = y0 + dim_input

        return self.grid[x0:x1, y0:y1]

    def get_samples(self, x0, x1, y0, y1, dim_input = 16*3*2-1):
        n_samples = (x1 - x0) * (y1-y0)

        samples = np.zeros( (n_samples, dim_input, dim_input) )
        coords = np.zeros( (n_samples, 2) )

        i = 0
        for xc in range(x0, x1):
            for yc in range(y0, y1):
                sample = self.get_sample(xc, yc, dim_input=dim_input)
                samples[i, :, :] = sample
                coords[i, :] = (xc, yc)
                i = i + 1

        return samples, coords
