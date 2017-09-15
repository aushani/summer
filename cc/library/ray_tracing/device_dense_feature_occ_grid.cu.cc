#include "library/ray_tracing/device_dense_feature_occ_grid.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"

namespace library {
namespace ray_tracing {

DeviceDenseFeatureOccGrid::DeviceDenseFeatureOccGrid(const Location *d_locs, const float *d_los, size_t sz_occ,
                            const Location *d_stats_locs, Stats *d_stats, size_t sz_stats,
                            float max_xy, float max_z, float res) :
 n_xy_(2*std::ceil(max_xy / res) + 1),
 n_z_(2*std::ceil(max_z / res) + 1),
 size_(n_xy_ * n_xy_ * n_z_),
 resolution_(res) {

  library::timer::Timer t;

  // Alloc
  cudaMalloc(&occu_, sizeof(bool)*size_);
  cudaMalloc(&known_, sizeof(bool)*size_);
  cudaMemset(known_, 0, sizeof(bool)*size_);

  cudaMalloc(&features_, sizeof(Feature)*size_);
  cudaMalloc(&has_features_, sizeof(bool)*size_);
  cudaMemset(has_features_, 0, sizeof(bool)*size_);

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  printf("Took %5.3f ms to alloc %ld MBytes for DDFOG\n",
      t.GetMs(), size_ * (sizeof(bool)*3 + sizeof(Feature))/(1024*1024));

  Populate(d_locs, d_los, sz_occ, d_stats_locs, d_stats, sz_stats);
}

void DeviceDenseFeatureOccGrid::Cleanup() {
  cudaFree(occu_);
  cudaFree(known_);

  cudaFree(features_);
  cudaFree(has_features_);
}

__global__ void PopulateDenseOcc(DeviceDenseFeatureOccGrid g, const Location *d_locs, const float *d_los, bool *occu, bool *known, size_t max) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= max) {
    return;
  }

  // Get index for the dense grid we're populating
  int idx_dense = g.GetIndex(d_locs[idx]);
  if (idx_dense < 0) {
    return;
  }

  // Now populate
  occu[idx_dense] = d_los[idx] > 0;
  known[idx_dense] = true;
}

__global__ void PopulateDenseOcc(DeviceDenseFeatureOccGrid g, const Location *d_locs, Stats *d_stats, Feature *features, bool *has_features, size_t max) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= max) {
    return;
  }

  // Get index for the dense grid we're populating
  int idx_dense = g.GetIndex(d_locs[idx]);
  if (idx_dense < 0) {
    return;
  }

  // TODO Blur?

  // Now populate
  Stats &s = d_stats[idx];
  features[idx_dense].theta = s.GetTheta();
  features[idx_dense].phi = s.GetPhi();
  has_features[idx_dense] = true;
}

void DeviceDenseFeatureOccGrid::Populate(const Location *d_locs, const float *d_los, size_t sz_occ,
                            const Location *d_stats_locs, Stats *d_stats, size_t sz_stats) {
  library::timer::Timer t;

  int threads = 1024;

  int blocks = std::ceil(sz_occ / static_cast<double>(threads));
  t.Start();
  PopulateDenseOcc<<<blocks, threads>>>((*this), d_locs, d_los, occu_, known_, sz_occ);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f ms to populate occ\n", t.GetMs());

  blocks = std::ceil(sz_stats / static_cast<double>(threads));
  t.Start();
  PopulateDenseOcc<<<blocks, threads>>>((*this), d_stats_locs, d_stats, features_, has_features_, sz_stats);
  err = cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorString(err));
  BOOST_ASSERT(err == cudaSuccess);
  printf("Took %5.3f ms to populate occ\n", t.GetMs());
}

float DeviceDenseFeatureOccGrid::GetResolution() const {
  return resolution_;
}

}  // namespace ray_tracing
}  // namespace library

