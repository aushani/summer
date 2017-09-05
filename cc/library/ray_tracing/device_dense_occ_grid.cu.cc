#include "library/ray_tracing/device_dense_occ_grid.h"

namespace library {
namespace ray_tracing {

DeviceDenseOccGrid::DeviceDenseOccGrid(const DeviceOccGrid &dog, float max_xy, float max_z) :
 n_xy_(2*std::ceil(max_xy / dog.GetResolution()) + 1),
 n_z_(2*std::ceil(max_z / dog.GetResolution()) + 1),
 size_(n_xy_ * n_xy_ * n_z_),
 resolution_(dog.GetResolution()) {

  cudaMalloc(&occu_, sizeof(bool)*size_);
  cudaMalloc(&known_, sizeof(bool)*size_);

  cudaMemset(&known_, 0, sizeof(bool)*size_);

  PopulateDense(dog);

}

DeviceDenseOccGrid::~DeviceDenseOccGrid() {
  cudaFree(occu_);
  cudaFree(known_);
}

__global__ void PopulateDenseKernel(const DeviceDenseOccGrid &ddog, const DeviceOccGrid &dog, bool *occu, bool *known) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= dog.size) {
    return;
  }

  int idx_dense = ddog.GetIndex(dog.locs[idx]);
  occu[idx_dense] = dog.los[idx] > 0;
  known[idx_dense] = false;
}

void DeviceDenseOccGrid::PopulateDense(const DeviceOccGrid &dog) {
  int threads = 1024;
  int blocks = std::ceil(dog.size / static_cast<double>(threads));

  PopulateDenseKernel<<<blocks, threads>>>((*this), dog, occu_, known_);
}

__device__ bool DeviceDenseOccGrid::IsKnown(const Location &loc) const {
  int idx = GetIndex(loc);
  if (idx < 0) {
    return false;
  }

  return known_[idx];
}

__device__ bool DeviceDenseOccGrid::IsOccu(const Location &loc) const {
  int idx = GetIndex(loc);
  if (idx < 0) {
    return false;
  }

  return occu_[idx];
}

__device__ int DeviceDenseOccGrid::GetIndex(const Location &loc) const {
  int ix = loc.i + n_xy_/2;
  int iy = loc.j + n_xy_/2;
  int iz = loc.k + n_z_/2;

  if (ix < 0 || ix >= n_xy_) {
    return -1;
  }

  if (iy < 0 || iy >= n_xy_) {
    return -1;
  }

  if (iz < 0 || iz >= n_z_) {
    return -1;
  }

  size_t idx = (ix*n_xy_ + iy)*n_z_ + iz;

  return idx;
}


}  // namespace ray_tracing
}  // namespace library
