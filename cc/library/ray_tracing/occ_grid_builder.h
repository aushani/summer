// Adapted from dascar
#pragma once

#include "occ_grid.h"

#include <memory>

#include <Eigen/Core>

namespace library {
namespace ray_tracing {

// Forward declaration, defined in occ_grid_builder.cu.cc
typedef struct DeviceData DeviceData;

// This class builds occupancy grids from given observations. Upon contruction,
// tt allocates resoucres on the GPU device.
class OccGridBuilder {
 public:
  // max_observations is the maximum number of ray tracing observations (i.e., points
  // from a velodyne scanner) that can be processed at a time. resolution determines the
  // resolution of the occpuancy grid generated. max_range is the maximum distance that
  // a ray tracing observation will sweep out free space in the occupancy grid.
  OccGridBuilder(int max_observations, float resolution, float max_range);
  ~OccGridBuilder();

  // Assumes that the hit origins are at (0, 0, 0)
  OccGrid GenerateOccGrid(const std::vector<Eigen::Vector3d> &hits);

 private:
  static const int kThreadsPerBlock_ = 1024;
  const float resolution_;
  const int max_observations_;

  const float kLogOddsFree_ = -0.1;
  const float kLogOddsOccupied_ = 1.0;
  const float kLogOddsUnknown_ = 0.0;

  std::unique_ptr<DeviceData> device_data_;
};

} // namespace ray_tracing
} // namespace library
