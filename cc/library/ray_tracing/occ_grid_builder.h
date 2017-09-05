// Adapted from dascar
#pragma once

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/device_occ_grid.h"

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

  // Configure the max dimension of outputted OccGrid's
  void ConfigureSize(float max_x, float max_y, float max_z);
  void ConfigureSizeInPixels(size_t max_i, size_t max_j, size_t max_k);

  void SetPose(const Eigen::Vector3d &pos, float theta);

  // Assumes that the hit origins are at (0, 0, 0)
  OccGrid GenerateOccGrid(const std::vector<Eigen::Vector3d> &hits);

  std::shared_ptr<DeviceOccGrid> GenerateOccGridDevice(const std::vector<Eigen::Vector3d> &hits);

 private:
  static constexpr float kLogOddsFree_ = -0.1;
  static constexpr float kLogOddsOccupied_ = 1.0;
  static constexpr float kLogOddsUnknown_ = 0.0;

  static constexpr int kThreadsPerBlock_ = 1024;
  const float resolution_;
  const int max_observations_;

  size_t max_i_ = 0;
  size_t max_j_ = 0;
  size_t max_k_ = 0;
  bool max_dimension_valid_ = false;

  std::unique_ptr<DeviceData> device_data_;
};

} // namespace ray_tracing
} // namespace library
