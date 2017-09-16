#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_dense_occ_grid.h"
#include "library/ray_tracing/device_feature_occ_grid.h"
#include "library/ray_tracing/device_dense_feature_occ_grid.h"
#include "library/ray_tracing/feature.h"

namespace library {
namespace ray_tracing {

class DeviceDenseFeatureOccGrid : public DeviceDenseOccGrid {
 public:
  DeviceDenseFeatureOccGrid(const DeviceFeatureOccGrid &dfog, float max_xy, float max_z);

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

  void Cleanup();

 private:
  Feature *features_ = nullptr;
  bool *has_features_ = nullptr;

  void PopulateFeatures(const DeviceFeatureOccGrid &dfog);
};

}  // namespace ray_tracing
}  // namespace library
