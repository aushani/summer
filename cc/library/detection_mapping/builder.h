#pragma once

#include "map.h"

#include <memory>

#include <Eigen/Core>

namespace library {
namespace detection_mapping {

// Forward declaration, defined in occ_grid_builder.cu.cc
typedef struct DeviceData DeviceData;

class Builder {
 public:
  Builder(double range_x, double range_y, float resolution, int max_observations, float max_range);
  ~Builder();

  // Assumes that the hit origins are at (0, 0, 0)
  //Map GenerateDetectionMap(const std::vector<Eigen::Vector3d> &hits);
  void GenerateDetectionMap(const std::vector<Eigen::Vector3d> &hits);

 private:
  static constexpr int kThreadsPerBlock_ = 1024;

  const float resolution_;
  const int max_observations_;

  std::unique_ptr<DeviceData> device_data_;
};

} // namespace ray_tracing
} // namespace library
