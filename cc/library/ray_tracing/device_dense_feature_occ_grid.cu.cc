#include "library/ray_tracing/device_dense_feature_occ_grid.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"

namespace library {
namespace ray_tracing {

DeviceDenseFeatureOccGrid::DeviceDenseFeatureOccGrid(const DeviceFeatureOccGrid &dfog, float max_xy, float max_z) :
 DeviceDenseOccGrid(dfog, max_xy, max_z) {
  // Alloc features
  cudaMalloc(&features_, sizeof(Feature)*size_);
  cudaMalloc(&has_features_, sizeof(bool)*size_);
  cudaMemset(has_features_, 0, sizeof(bool)*size_);

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  printf("Alloced %ld MBytes for DDFOG\n", size_ * (sizeof(bool)*3 + sizeof(Feature))/(1024*1024));

  PopulateFeatures(dfog);
}

void DeviceDenseFeatureOccGrid::Cleanup() {
  DeviceDenseOccGrid::Cleanup();

  cudaFree(features_);
  cudaFree(has_features_);
}

__global__ void PopulateFeaturesKernel(DeviceDenseFeatureOccGrid g, const DeviceFeatureOccGrid dfog, Feature *features, bool *has_features) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= dfog.sz_features) {
    return;
  }

  // Get index for the dense grid we're populating
  const Location &loc = dfog.feature_locs[idx];
  int idx_dense = g.GetIndex(loc);
  if (idx_dense < 0) {
    return;
  }

  // Now populate
  features[idx_dense] = dfog.features[idx];
  has_features[idx_dense] = true;
}

void DeviceDenseFeatureOccGrid::PopulateFeatures(const DeviceFeatureOccGrid &dfog) {
  library::timer::Timer t;

  int threads = 1024;

  int blocks = std::ceil(dfog.sz_features / static_cast<double>(threads));
  t.Start();
  PopulateFeaturesKernel<<<blocks, threads>>>((*this), dfog, features_, has_features_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f ms to populate stats with %ld elements\n", t.GetMs(), dfog.sz_features);
}

}  // namespace ray_tracing
}  // namespace library

