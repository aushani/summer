#pragma once

#include <Eigen/Core>

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/device_feature_occ_grid.h"
#include "library/ray_tracing/feature.h"

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

class FeatureOccGrid : public OccGrid {
 public:
  static FeatureOccGrid FromDevice(const DeviceFeatureOccGrid &dfog);

  const std::vector<Location>& GetFeatureLocations() const;
  const std::vector<Feature>& GetFeatures() const;

  bool HasFeature(const Location &loc) const;

  const Feature& GetFeature(const Location &loc) const;
  Eigen::Vector3f GetNormal(const Location &loc) const;

 private:
  // Parallel containers
  std::vector<Location> feature_locs_;
  std::vector<Feature> features_;

  // For convience
  FeatureOccGrid() : OccGrid() {}

  size_t GetFeaturePos(const Location &loc) const;
};

}  // namespace ray_tracing
}  // namespace library
