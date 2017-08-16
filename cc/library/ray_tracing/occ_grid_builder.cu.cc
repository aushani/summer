// Adapted from dascar
#include "library/ray_tracing/occ_grid_builder.h"

#include <iostream>

#include <boost/assert.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

// This is the data we need to generate occupancy grids that
// is passed to the device. Package it in this way so we can
// take advantage of coalesced memory operations on the GPU
// for faster performance.
struct DeviceData {
  DeviceData(float resolution, float max_range, int max_observations,
      float logOddsFree, float logOddsOccupied, float logOddsUnknown);
  DeviceData(const DeviceData &dd);
  ~DeviceData();

  void CopyData(const std::vector<Eigen::Vector3d> &hits);

  int num_observations = 0;
  int max_voxel_visits_per_ray = 0;

  float resolution = 0.0f;

  float *hit_x = nullptr;
  float *hit_y = nullptr;
  float *hit_z = nullptr;

  float *origin_x = nullptr;
  float *origin_y = nullptr;
  float *origin_z = nullptr;

  Location *locations = nullptr;
  float *log_odds_updates = nullptr;

  Location *locations_reduced = nullptr;
  float *log_odds_updates_reduced = nullptr;

  bool own_gpu_memory = false;

  const float kLogOddsFree;
  const float kLogOddsOccupied;
  const float kLogOddsUnknown;
};

DeviceData::DeviceData(float resolution, float max_range, int max_observations,
    float logOddsFree, float logOddsOccupied, float logOddsUnknown)
    : resolution(resolution), max_voxel_visits_per_ray(max_range / resolution),
    kLogOddsFree(logOddsFree), kLogOddsOccupied(logOddsOccupied), kLogOddsUnknown(logOddsUnknown) {
  // Allocate memory on the device.
  cudaError_t err = cudaMalloc(&hit_x, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_y, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_z, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&origin_x, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&origin_y, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&origin_z, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&locations, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&locations_reduced, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates_reduced, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);
}

DeviceData::DeviceData(const DeviceData &dd)
    : num_observations(dd.num_observations),
      max_voxel_visits_per_ray(dd.max_voxel_visits_per_ray),
      resolution(dd.resolution),
      hit_x(dd.hit_x),
      hit_y(dd.hit_y),
      hit_z(dd.hit_z),
      origin_x(dd.origin_x),
      origin_y(dd.origin_y),
      origin_z(dd.origin_z),
      locations(dd.locations),
      log_odds_updates(dd.log_odds_updates),
      locations_reduced(dd.locations_reduced),
      log_odds_updates_reduced(dd.log_odds_updates_reduced),
      own_gpu_memory(false),
      kLogOddsFree(dd.kLogOddsFree),
      kLogOddsOccupied(dd.kLogOddsOccupied),
      kLogOddsUnknown(dd.kLogOddsUnknown)
      {}

DeviceData::~DeviceData() {
  if (own_gpu_memory) {
    cudaFree(hit_x);
    cudaFree(hit_y);
    cudaFree(hit_z);
    cudaFree(origin_x);
    cudaFree(origin_y);
    cudaFree(origin_z);
    cudaFree(locations);
    cudaFree(log_odds_updates);
    cudaFree(locations_reduced);
    cudaFree(log_odds_updates_reduced);
  }
}

void DeviceData::CopyData(const std::vector<Eigen::Vector3d> &hits) {
  num_observations = hits.size();

  std::vector<float> v_hit_x, v_hit_y, v_hit_z;
  std::vector<float> v_origin_x, v_origin_y, v_origin_z;
  for (const auto &hit : hits) {
    v_hit_x.push_back(hit.x());
    v_hit_y.push_back(hit.y());
    v_hit_z.push_back(hit.z());

    // Assumes that the hits originated from (0, 0, 0)
    v_origin_x.push_back(0);
    v_origin_y.push_back(0);
    v_origin_z.push_back(0);
  }

  cudaStream_t streams[6];
  for (int i = 0; i < 6; i++) {
    cudaError_t res = cudaStreamCreate(&streams[i]);
    BOOST_ASSERT(res == cudaSuccess);
  }

  size_t sz_copy = sizeof(float) * hits.size();
  cudaMemcpyAsync(hit_x,    v_hit_x.data(),    sz_copy, cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(hit_y,    v_hit_y.data(),    sz_copy, cudaMemcpyHostToDevice, streams[1]);
  cudaMemcpyAsync(hit_z,    v_hit_z.data(),    sz_copy, cudaMemcpyHostToDevice, streams[2]);
  cudaMemcpyAsync(origin_x, v_origin_x.data(), sz_copy, cudaMemcpyHostToDevice, streams[3]);
  cudaMemcpyAsync(origin_y, v_origin_y.data(), sz_copy, cudaMemcpyHostToDevice, streams[4]);
  cudaMemcpyAsync(origin_z, v_origin_z.data(), sz_copy, cudaMemcpyHostToDevice, streams[5]);

  // Wait for all memcpy streams to complete
  cudaDeviceSynchronize();

  for (int i = 0; i < 6; i++) {
    cudaError_t res = cudaStreamDestroy(streams[i]);
    BOOST_ASSERT(res == cudaSuccess);
  }
}

OccGridBuilder::OccGridBuilder(int max_observations, float resolution, float max_range)
    : max_observations_(max_observations),
      resolution_(resolution),
      device_data_(new DeviceData(resolution, max_range, max_observations,
            kLogOddsFree_, kLogOddsOccupied_, kLogOddsUnknown_)) {
  device_data_->own_gpu_memory = true;
}

OccGridBuilder::~OccGridBuilder() {}

__global__ void RayTracingKernel(DeviceData data) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int hit_idx = tidx + bidx * threads;
  if (hit_idx >= data.num_observations) {
    return;
  }

  float hit[3] = {data.hit_x[hit_idx], data.hit_y[hit_idx], data.hit_z[hit_idx]};
  float origin[3] = {data.origin_x[hit_idx], data.origin_y[hit_idx], data.origin_z[hit_idx]};

  // The following is an implementation of Bresenham's line algorithm to sweep out the ray from the origin of the ray to
  // the hit point.
  // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
  float ad[3] = {0.0f, 0.0f, 0.0f};  // absolute value of diff * 2
  int sgn[3] = {0, 0, 0};            // which way am i going
  int cur_loc[3] = {0, 0, 0};        // what location am i currently at (starts at origin of ray)
  int end_loc[3] = {0, 0, 0};        // what location am i ending at (ends at the hit)
  int dominant_dim = 0;              // which dim am i stepping through

  for (int i = 0; i < 3; ++i) {
    cur_loc[i] = round(origin[i] / data.resolution);
    end_loc[i] = round(hit[i] / data.resolution);

    ad[i] = fabsf(end_loc[i] - cur_loc[i]) * 2;

    sgn[i] = (end_loc[i] > cur_loc[i]) - (cur_loc[i] > end_loc[i]);

    if (ad[i] > ad[dominant_dim]) {
      dominant_dim = i;
    }
  }

  float err[3];
  for (int i = 0; i < 3; i++) {
    err[i] = ad[i] - ad[dominant_dim] / 2;
  }

  // walk down ray
  size_t mem_step_size = data.num_observations;
  size_t mem_idx = hit_idx;
  bool valid = true;
  for (int step = 0; step < (data.max_voxel_visits_per_ray - 1); ++step) {
    Location loc(cur_loc[0], cur_loc[1], cur_loc[2]);
    float loUpdate = data.kLogOddsUnknown;

    // Are we done? Have we reached the hit point?
    // Don't quit the loop just yet. We need to 0 out the rest of log odds updates.
    if ((sgn[dominant_dim] > 0) ? (cur_loc[dominant_dim] >= end_loc[dominant_dim])
                                : (cur_loc[dominant_dim] <= end_loc[dominant_dim])) {
      valid = false;
    }

    if (valid) {
      // step forwards
      for (int dim = 0; dim < 3; ++dim) {
        if (dim != dominant_dim) {
          if (err[dim] >= 0) {
            cur_loc[dim] += sgn[dim];
            err[dim] -= ad[dominant_dim];
          }
        }
      }

      for (int dim = 0; dim < 3; ++dim) {
        if (dim == dominant_dim) {
          cur_loc[dim] += sgn[dim];
        } else {
          err[dim] += ad[dim];
        }
      }

      loUpdate = data.kLogOddsFree;
    }

    // Now write out key value pair
    data.locations[mem_idx] = loc;
    data.log_odds_updates[mem_idx] = loUpdate;

    mem_idx += mem_step_size;
  }

  Location loc(end_loc[0], end_loc[1], end_loc[2]);

  // Now write out key value pair
  data.locations[mem_idx] = loc;
  data.log_odds_updates[mem_idx] = data.kLogOddsOccupied;
}

struct NoUpdate {
  __host__ __device__ bool operator()(const float x) const { return fabs(x) < 1e-6; }
};

OccGrid OccGridBuilder::GenerateOccGrid(const std::vector<Eigen::Vector3d> &hits) {
  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    std::vector<Location> location_vector;
    std::vector<float> lo_vector;
    return OccGrid(location_vector, lo_vector, resolution_);
  }

  // First, we need to send the data to the GPU device
  device_data_->CopyData(hits);

  // Now run ray tracing on the GPU device
  int blocks = ceil(static_cast<float>(device_data_->num_observations) / kThreadsPerBlock_);
  RayTracingKernel<<<blocks, kThreadsPerBlock_>>>(*device_data_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  // Accumulate all the updates
  size_t num_updates = device_data_->num_observations * device_data_->max_voxel_visits_per_ray;

  // First prune unnecessary updates
  thrust::device_ptr<Location> dp_locations(device_data_->locations);
  thrust::device_ptr<float> dp_updates(device_data_->log_odds_updates);

  thrust::device_ptr<Location> dp_locations_end =
      thrust::remove_if(dp_locations, dp_locations + num_updates, dp_updates, NoUpdate());
  thrust::device_ptr<float> dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, NoUpdate());
  num_updates = dp_locations_end - dp_locations;

  // Now reduce updates to resulting log odds
  thrust::sort_by_key(dp_locations, dp_locations + num_updates, dp_updates);

  thrust::device_ptr<Location> dp_locs_reduced(device_data_->locations_reduced);
  thrust::device_ptr<float> dp_lo_reduced(device_data_->log_odds_updates_reduced);

  thrust::pair<thrust::device_ptr<Location>, thrust::device_ptr<float> > new_ends = thrust::reduce_by_key(
      dp_locations, dp_locations + num_updates, dp_updates, dp_locs_reduced, dp_lo_reduced);
  num_updates = new_ends.first - dp_locs_reduced;

  // Copy result from GPU device to host
  std::vector<Location> location_vector(num_updates);
  std::vector<float> lo_vector(num_updates);
  cudaMemcpy(location_vector.data(), dp_locs_reduced.get(), sizeof(Location) * num_updates, cudaMemcpyDeviceToHost);
  cudaMemcpy(lo_vector.data(), dp_lo_reduced.get(), sizeof(float) * num_updates, cudaMemcpyDeviceToHost);

  return OccGrid(location_vector, lo_vector, resolution_);
}

}  // namespace ray_tracing
}  // namespace library
