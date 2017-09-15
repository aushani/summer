#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/stats.h"

namespace library {
namespace ray_tracing {

struct Feature {
  float theta = 0;
  float phi = 0;
};

class DeviceDenseFeatureOccGrid {
 public:
  DeviceDenseFeatureOccGrid(const Location *d_locs, const float *d_los, size_t sz_occ,
                            const Location *d_stats_locs, Stats *d_stats, size_t sz_stats,
                            float max_xy, float max_z, float res);

  void Cleanup();
  float GetResolution() const;

  __device__ bool IsKnown(const Location &loc) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return false;
    }

    return known_[idx];
  }

  __device__ bool IsOccu(const Location &loc) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return false;
    }

    return occu_[idx];
  }

  __device__ bool HasFeature(const Location &loc) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return false;
    }

    return has_features_[idx];
  }

  __device__ Feature GetFeature(const Location &loc) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return Feature();
    }

    return features_[idx];
  }

  __device__ int GetIndex(const Location &loc) const {
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

 private:
  bool *occu_ = nullptr;
  bool *known_ = nullptr;

  Feature *features_ = nullptr;
  bool *has_features_ = nullptr;

  int n_xy_ = 0;
  int n_z_ = 0;
  int size_ = 0;

  float resolution_ = 0;

  void Populate(const Location *d_locs, const float *d_los, size_t sz_occ,
      const Location *d_stats_locs, Stats *d_stats, size_t sz_stats);
};

}  // namespace ray_tracing
}  // namespace library
