#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_occ_grid.h"

namespace library {
namespace ray_tracing {

class DeviceDenseOccGrid {
 public:
  DeviceDenseOccGrid(const DeviceOccGrid &dog, float max_xy, float max_z);

  void Cleanup();

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

  int n_xy_ = 0;
  int n_z_ = 0;
  int size_ = 0;

  float resolution_ = 0;

  void PopulateDense(const DeviceOccGrid &dog);
};

}  // namespace ray_tracing
}  // namespace library
