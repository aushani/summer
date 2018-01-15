// Adapted from dascar
#include "library/ray_tracing/occ_grid_builder.h"

#include <iostream>

#include <boost/assert.hpp>

#include <thrust/device_ptr.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/ray_tracing/device_data.cu.h"
#include "library/timer/timer.h"

namespace tr = library::timer;

namespace library {
namespace ray_tracing {

OccGridBuilder::OccGridBuilder(int max_observations, float resolution, float max_range)
    : max_observations_(max_observations),
      resolution_(resolution),
      device_data_(new DeviceData(resolution, max_range, max_observations)) {
}

OccGridBuilder::~OccGridBuilder() {
  device_data_->FreeDeviceMemory();
}

void OccGridBuilder::ConfigureSize(float max_x, float max_y, float max_z) {
  ConfigureSizeInPixels(ceil(max_x / resolution_), ceil(max_y / resolution_), ceil(max_z / resolution_));
}

void OccGridBuilder::ConfigureSizeInPixels(size_t max_i, size_t max_j, size_t max_k) {
  device_data_->oor = OutOfRange(max_i, max_j, max_k);
  device_data_->oor_valid = true;

  int max_ijk = max_i;
  if (max_ijk < max_j ) max_ijk = max_j;
  if (max_ijk < max_k ) max_ijk = max_k;

  // From -max to +max
  max_ijk *= 2;

  device_data_->steps_per_ray = max_ijk;
  if (max_ijk > device_data_->max_voxel_visits_per_ray) {
    device_data_->steps_per_ray = device_data_->max_voxel_visits_per_ray;
  }
}

void OccGridBuilder::SetPose(const Eigen::Vector3d &pos, float theta) {
  device_data_->pose_xyz[0] = pos.x();
  device_data_->pose_xyz[1] = pos.y();
  device_data_->pose_xyz[2] = pos.z();

  device_data_->pose_theta = theta;
}

size_t OccGridBuilder::ProcessData(const std::vector<Eigen::Vector3d> &hits) {
  // First, we need to send the data to the GPU device
  device_data_->CopyData(hits);

  // Now run ray tracing on the GPU device
  device_data_->RunKernel(false);
  return device_data_->ReduceLogOdds();
}

OccGrid OccGridBuilder::GenerateOccGrid(const std::vector<Eigen::Vector3d> &hits) {
  //library::timer::Timer t;

  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    std::vector<Location> location_vector;
    std::vector<float> lo_vector;
    return OccGrid(location_vector, lo_vector, resolution_);
  }

  size_t num_updates = ProcessData(hits);

  // Copy result from GPU device to host
  //t.Start();
  std::vector<Location> location_vector(num_updates);
  std::vector<float> lo_vector(num_updates);
  cudaMemcpy(location_vector.data(), device_data_->locations_reduced, sizeof(Location) * num_updates, cudaMemcpyDeviceToHost);
  cudaMemcpy(lo_vector.data(), device_data_->log_odds_updates_reduced, sizeof(float) * num_updates, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  //printf("\tTook %5.3f to copy to host\n", t.GetMs());

  return OccGrid(location_vector, lo_vector, resolution_);
}

std::shared_ptr<DeviceOccGrid> OccGridBuilder::GenerateOccGridDevice(const std::vector<Eigen::Vector3d> &hits) {
  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    std::vector<Location> location_vector;
    std::vector<float> lo_vector;
    return nullptr;
  }

  size_t num_updates = ProcessData(hits);

  return std::make_shared<DeviceOccGrid>(device_data_->locations_reduced, device_data_->log_odds_updates_reduced , num_updates, resolution_);
}

DeviceFeatureOccGrid OccGridBuilder::GenerateDeviceFeatureOccGrid(const std::vector<Eigen::Vector3d> &hits, const std::vector<float> &intensity) {
  // Send data
  device_data_->CopyData(hits);
  device_data_->CopyIntensity(intensity);

  device_data_->RunKernel(true);

  size_t num_updates = device_data_->ReduceLogOdds();
  size_t num_stats = device_data_->ReduceStats();
  //printf("Got %d stats\n", num_stats);

  return DeviceFeatureOccGrid(device_data_->locations_reduced, device_data_->log_odds_updates_reduced, num_updates,
                              device_data_->stats_locs_reduced, device_data_->stats_reduced, num_stats, resolution_);
}

FeatureOccGrid OccGridBuilder::GenerateFeatureOccGrid(const std::vector<Eigen::Vector3d> &hits, const std::vector<float> &intensity) {
  auto dfog = GenerateDeviceFeatureOccGrid(hits, intensity);
  auto fog = FeatureOccGrid::FromDevice(dfog);

  dfog.Cleanup();

  return fog;
}

}  // namespace ray_tracing
}  // namespace library
