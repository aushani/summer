#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

class DeviceOccGrid {
 public:
  DeviceOccGrid(const thrust::device_ptr<Location> &d_locs, const thrust::device_ptr<float> &d_los, size_t sz, float r);

  // Need a seperate cleanup rather than destructor because of how this is passed to CUDA kernels
  void Cleanup();

  float GetResolution() const;

  Location *locs = nullptr;
  float *los = nullptr;
  int size = 0;
  float res = 0;
};

}  // namespace ray_tracing
}  // namespace library
