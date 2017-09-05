#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_occ_grid.h"

namespace library {
namespace ray_tracing {

class DeviceDenseOccGrid {
 public:
  DeviceDenseOccGrid(const DeviceOccGrid &dog, float max_xy, float max_z);
  DeviceDenseOccGrid(const DeviceDenseOccGrid &dog) = delete;
  ~DeviceDenseOccGrid();

  __device__ bool IsKnown(const Location &loc) const;
  __device__ bool IsOccu(const Location &loc) const;

  __device__ int GetIndex(const Location &loc) const;

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
