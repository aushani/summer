#include "library/ray_tracing/device_feature_occ_grid.h"

#include <math.h>
#include <boost/assert.hpp>

#include "library/timer/timer.h"

namespace library {
namespace ray_tracing {

DeviceFeatureOccGrid::DeviceFeatureOccGrid(const Location *d_locs, const float *d_los, size_t sz_occ,
                     const Location *d_stats_locs, const Stats *d_stats, size_t sz_stats, float res) :
 DeviceOccGrid(d_locs, d_los, sz_occ, res), sz_features(sz_stats) {
  // Features
  cudaMalloc(&feature_locs, sizeof(Location) * sz_stats);
  cudaMalloc(&features, sizeof(Feature) * sz_stats);

  PopulateFeatures(d_stats_locs, d_stats);
}

void DeviceFeatureOccGrid::Cleanup() {
  DeviceOccGrid::Cleanup();

  cudaFree(feature_locs);
  cudaFree(features);
}

// Essentially a binary search
__device__ int GetIndex(const Location *locs, const Location &query, int first, int last) {

  // Base cases
  if (first >= last) {
    return -1;
  }

  if (first + 1 == last) {
    const Location &loc_at = locs[first];
    if (loc_at == query) {
      return first;
    } else {
      return -1;
    }
  }

  int mid = (first + last) / 2;
  const Location &loc_mid = locs[mid];
  if (loc_mid == query) {
    return mid;
  }

  if (loc_mid < query) {
    first = mid + 1;
  } else {
    last = mid;
  }

  // Recurse
  return GetIndex(locs, query, first, last);
}

__global__ void ComputeFeatures(DeviceFeatureOccGrid dfog, const Location *d_locs, const Stats *d_stats) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= dfog.sz_features) {
    return;
  }

  // Get the loc we're processing
  const Location &loc = d_locs[idx];

  // TODO Blur?
  const int blur_size = 1;
  Stats blurred_stats;

  for (int di=-blur_size; di<=blur_size; di++) {
    for (int dj=-blur_size; dj<=blur_size; dj++) {
      for (int dk=-blur_size; dk<=blur_size; dk++) {
        Location loc_i(loc.i + di, loc.j + dj, loc.k + dk);

        int idx_loc = GetIndex(d_locs, loc_i, 0, dfog.sz_features);

        if (idx_loc >= 0) {
          blurred_stats = blurred_stats + d_stats[idx_loc];
        }
      }
    }
  }
  // DEBUG Stats blurred_stats = d_stats[idx];

  // Now populate
  dfog.feature_locs[idx] = loc;

  bool valid_normal = blurred_stats.count > 3;
  dfog.features[idx] = Feature(blurred_stats.GetTheta(), blurred_stats.GetPhi(), valid_normal, blurred_stats.intensity);

  // DEBUG
  //float x = loc.i;
  //float y = loc.j;
  //float z = loc.k;

  //float d_xy = sqrt(x*x + y*y + z*z);
  //dfog.features[idx].theta = atan2(y, x);
  //dfog.features[idx].phi = atan2(z, d_xy);
}

void DeviceFeatureOccGrid::PopulateFeatures(const Location *d_stats_locs, const Stats *d_stats) {
  //library::timer::Timer t;

  int threads = 128;
  int blocks = std::ceil(sz_features / static_cast<double>(threads));

  //t.Start();
  ComputeFeatures<<<blocks, threads>>>((*this), d_stats_locs, d_stats);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  //printf("Took %5.3f ms to populate stats with %ld elements with %d threads and %d blocks\n",
  //    t.GetMs(), sz_features, threads, blocks);

}

}  // namespace ray_tracing
}  // namespace library

