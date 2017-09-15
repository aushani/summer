#pragma once

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/stats.h"
#include "library/ray_tracing/feature.h"

namespace library {
namespace ray_tracing {

class DeviceFeatureOccGrid : public DeviceOccGrid {
 public:
  DeviceFeatureOccGrid(const Location *d_locs, const float *d_los, size_t sz_occ,
                       const Location *d_stats_locs, const Stats *d_stats, size_t sz_stats, float res);

  void Cleanup();

  Location *feature_locs = nullptr;
  Feature *features = nullptr;
  size_t sz_features = 0;

 private:
  void PopulateFeatures(const Location *d_stats_locs, const Stats *d_stats);
};

}  // namespace ray_tracing
}  // namespace library
