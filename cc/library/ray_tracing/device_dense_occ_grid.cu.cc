#include "library/ray_tracing/device_dense_occ_grid.h"

#include <boost/assert.hpp>

namespace library {
namespace ray_tracing {

DeviceDenseOccGrid::DeviceDenseOccGrid(const DeviceOccGrid &dog, float max_xy, float max_z) :
 n_xy_(2*std::ceil(max_xy / dog.GetResolution()) + 1),
 n_z_(2*std::ceil(max_z / dog.GetResolution()) + 1),
 size_(n_xy_ * n_xy_ * n_z_),
 resolution_(dog.GetResolution()) {

  cudaMalloc(&occu_, sizeof(bool)*size_);
  cudaMalloc(&known_, sizeof(bool)*size_);

  cudaMemset(known_, 0, sizeof(bool)*size_);

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  PopulateDense(dog);
}

void DeviceDenseOccGrid::Cleanup() {
  cudaFree(occu_);
  cudaFree(known_);
}

__global__ void PopulateDenseKernel(const DeviceDenseOccGrid ddog, const DeviceOccGrid dog, bool *occu, bool *known) {
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
  known[idx_dense] = true;
}

void DeviceDenseOccGrid::PopulateDense(const DeviceOccGrid &dog) {
  int threads = 1024;
  int blocks = std::ceil(dog.size / static_cast<double>(threads));

  PopulateDenseKernel<<<blocks, threads>>>((*this), dog, occu_, known_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
}

}  // namespace ray_tracing
}  // namespace library
