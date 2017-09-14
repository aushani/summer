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
  max_i_ = max_i;
  max_j_ = max_j;
  max_k_ = max_k;
  max_dimension_valid_ = true;
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

  if (max_dimension_valid_) {
    OutOfRange oor(max_i_, max_j_, max_k_);
    return device_data_->ReduceLogOdds(oor);
  } else {
    return device_data_->ReduceLogOdds();
  }
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

FeatureOccGrid OccGridBuilder::GenerateFeatureOccGrid(const std::vector<Eigen::Vector3d> &hits, const std::vector<float> &intensity) {
  // Send data
  device_data_->CopyData(hits);
  device_data_->CopyIntensity(intensity);

  device_data_->RunKernel(true);

  size_t num_updates = device_data_->ReduceLogOdds();
  size_t num_stats = device_data_->ReduceStats();
  //printf("Got %d stats\n", num_stats);

  // Copy results back
  std::vector<Location> location_vector(num_updates);
  std::vector<float> lo_vector(num_updates);
  cudaMemcpy(location_vector.data(), device_data_->locations_reduced, sizeof(Location) * num_updates, cudaMemcpyDeviceToHost);
  cudaMemcpy(lo_vector.data(), device_data_->log_odds_updates_reduced, sizeof(float) * num_updates, cudaMemcpyDeviceToHost);

  std::vector<Location> sl_vector(num_stats);
  std::vector<Stats> s_vector(num_stats);
  cudaMemcpy(sl_vector.data(), device_data_->stats_locs_reduced, sizeof(Location) * num_stats, cudaMemcpyDeviceToHost);
  cudaMemcpy(s_vector.data(), device_data_->stats_reduced, sizeof(Stats) * num_stats, cudaMemcpyDeviceToHost);

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  return FeatureOccGrid(location_vector, lo_vector, sl_vector, s_vector, resolution_);
  //return FeatureOccGrid(location_vector, lo_vector, resolution_);
}

}  // namespace ray_tracing
}  // namespace library
