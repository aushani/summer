// Adapted from dascar
#include "library/ray_tracing/occ_grid_builder.h"

#include <iostream>

#include <boost/assert.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include "library/ray_tracing/occ_grid_location.h"
#include "library/timer/timer.h"

namespace tr = library::timer;

namespace library {
namespace ray_tracing {

// This is the data we need to generate occupancy grids that
// is passed to the device. Package it in this way so we can
// take advantage of coalesced memory operations on the GPU
// for faster performance.
struct DeviceData {
  DeviceData(float resolution, float max_range, int max_observations,
      float logOddsFree, float logOddsOccupied, float logOddsUnknown);

  void FreeDeviceMemory();

  void CopyData(const std::vector<Eigen::Vector3d> &hits);

  int num_observations = 0;
  int max_voxel_visits_per_ray = 0;

  float resolution = 0.0f;

  float *hit_x = nullptr;
  float *hit_y = nullptr;
  float *hit_z = nullptr;

  Location *locations = nullptr;
  float *log_odds_updates = nullptr;

  Location *locations_reduced = nullptr;
  float *log_odds_updates_reduced = nullptr;

  float pose_xyz[3] = {0.0f, 0.0f, 0.0f};
  float pose_theta = 0.0f;

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

  err = cudaMalloc(&locations, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&locations_reduced, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates_reduced, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  BOOST_ASSERT(err == cudaSuccess);
}

void DeviceData::FreeDeviceMemory() {
  cudaFree(hit_x);
  cudaFree(hit_y);
  cudaFree(hit_z);
  cudaFree(locations);
  cudaFree(log_odds_updates);
  cudaFree(locations_reduced);
  cudaFree(log_odds_updates_reduced);
}

void DeviceData::CopyData(const std::vector<Eigen::Vector3d> &hits) {
  num_observations = hits.size();

  std::vector<float> v_hit_x, v_hit_y, v_hit_z;
  for (const auto &hit : hits) {
    auto d = hit.data();
    v_hit_x.push_back(d[0]);
    v_hit_y.push_back(d[1]);
    v_hit_z.push_back(d[2]);

    // Assumes that the hits originated from (0, 0, 0)
  }

  size_t sz_copy = sizeof(float) * hits.size();

  //cudaStream_t streams[3];
  //for (int i = 0; i < 3; i++) {
  //  cudaError_t res = cudaStreamCreate(&streams[i]);
  //  BOOST_ASSERT(res == cudaSuccess);
  //}

  //cudaMemcpyAsync(hit_x,    v_hit_x.data(),    sz_copy, cudaMemcpyHostToDevice, streams[0]);
  //cudaMemcpyAsync(hit_y,    v_hit_y.data(),    sz_copy, cudaMemcpyHostToDevice, streams[1]);
  //cudaMemcpyAsync(hit_z,    v_hit_z.data(),    sz_copy, cudaMemcpyHostToDevice, streams[2]);

  //// Wait for all memcpy streams to complete
  //cudaDeviceSynchronize();

  //for (int i = 0; i < 6; i++) {
  //  cudaError_t res = cudaStreamDestroy(streams[i]);
  //  BOOST_ASSERT(res == cudaSuccess);
  //}

  cudaMemcpy(hit_x,    v_hit_x.data(),    sz_copy, cudaMemcpyHostToDevice);
  cudaMemcpy(hit_y,    v_hit_y.data(),    sz_copy, cudaMemcpyHostToDevice);
  cudaMemcpy(hit_z,    v_hit_z.data(),    sz_copy, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

OccGridBuilder::OccGridBuilder(int max_observations, float resolution, float max_range)
    : max_observations_(max_observations),
      resolution_(resolution),
      device_data_(new DeviceData(resolution, max_range, max_observations,
            kLogOddsFree_, kLogOddsOccupied_, kLogOddsUnknown_)) {
}

OccGridBuilder::~OccGridBuilder() {
  device_data_->FreeDeviceMemory();
}

__global__ void RayTracingKernel(DeviceData data) {
  // Figure out which hit this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int hit_idx = tidx + bidx * threads;
  if (hit_idx >= data.num_observations) {
    return;
  }

  // Get origin and hit relative to pose
  float hx = data.hit_x[hit_idx] - data.pose_xyz[0];
  float hy = data.hit_y[hit_idx] - data.pose_xyz[1];
  float hz = data.hit_z[hit_idx] - data.pose_xyz[2];

  float st = sin(data.pose_theta);
  float ct = cos(data.pose_theta);

  float hx_p = ct * hx - st * hy;
  float hy_p = st * hx + ct * hy;
  float hz_p = hz;

  float ox = -data.pose_xyz[0];
  float oy = -data.pose_xyz[1];
  float oz = -data.pose_xyz[2];

  float ox_p = ct * ox - st * oy;
  float oy_p = st * ox + ct * oy;
  float oz_p = oz;

  float hit[3] = {hx_p, hy_p, hz_p};
  float origin[3] = {ox_p, oy_p, oz_p};

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

struct OutOfRange {
  int max_i;
  int max_j;
  int max_k;

  OutOfRange(float max_x, float max_y, float max_z, float resolution) {
    max_i = ceil(max_x / resolution);
    max_j = ceil(max_y / resolution);
    max_k = ceil(max_z / resolution);
  }

  __host__ __device__ bool operator()(const Location &loc) const {
    if (std::abs(loc.i) >= max_i) {
      return true;
    }
    if (std::abs(loc.j) >= max_j) {
      return true;
    }
    if (std::abs(loc.k) >= max_k) {
      return true;
    }

    return false;
  }
};

void OccGridBuilder::ConfigureSize(float max_x, float max_y, float max_z) {
  max_x_ = max_x;
  max_y_ = max_y;
  max_z_ = max_z;
  max_dimension_valid_ = true;
}

void OccGridBuilder::SetPose(const Eigen::Vector3d &pos, float theta) {
  device_data_->pose_xyz[0] = pos.x();
  device_data_->pose_xyz[1] = pos.y();
  device_data_->pose_xyz[2] = pos.z();

  device_data_->pose_theta = theta;
}

OccGrid OccGridBuilder::GenerateOccGrid(const std::vector<Eigen::Vector3d> &hits) {
  library::timer::Timer t;

  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    std::vector<Location> location_vector;
    std::vector<float> lo_vector;
    return OccGrid(location_vector, lo_vector, resolution_);
  }

  // First, we need to send the data to the GPU device
  t.Start();
  device_data_->CopyData(hits);
  printf("\tTook %5.3f ms to copy data\n", t.GetMs());

  // Now run ray tracing on the GPU device
  t.Start();
  int blocks = ceil(static_cast<float>(device_data_->num_observations) / kThreadsPerBlock_);
  RayTracingKernel<<<blocks, kThreadsPerBlock_>>>(*device_data_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tTook %5.3f ms to run kernel\n", t.GetMs());

  // Accumulate all the updates
  size_t num_updates = device_data_->num_observations * device_data_->max_voxel_visits_per_ray;

  // First prune unnecessary updates
  t.Start();
  thrust::device_ptr<Location> dp_locations(device_data_->locations);
  thrust::device_ptr<float> dp_updates(device_data_->log_odds_updates);

  auto dp_locations_end = thrust::remove_if(dp_locations, dp_locations + num_updates, dp_updates, NoUpdate());
  auto dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, NoUpdate());
  printf("\tTook %5.3f to prune from %ld to %ld\n", t.GetMs(), num_updates, dp_locations_end - dp_locations);
  num_updates = dp_locations_end - dp_locations;

  // Prune updates that are out of range
  if (max_dimension_valid_) {
    t.Start();
    OutOfRange oor(max_x_, max_y_, max_z_, resolution_);
    auto dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, dp_locations, oor);
    auto dp_locations_end = thrust::remove_if(dp_locations, dp_locations + num_updates, oor);
    printf("\tTook %5.3f to enfore in range (%ld->%ld)\n",
        t.GetMs(), num_updates, dp_locations_end - dp_locations);
    num_updates = dp_locations_end - dp_locations;
  }

  // Now reduce updates to resulting log odds
  t.Start();
  thrust::sort_by_key(dp_locations, dp_locations + num_updates, dp_updates);
  printf("\tTook %5.3f to sort\n", t.GetMs());

  t.Start();
  thrust::device_ptr<Location> dp_locs_reduced(device_data_->locations_reduced);
  thrust::device_ptr<float> dp_lo_reduced(device_data_->log_odds_updates_reduced);

  thrust::pair<thrust::device_ptr<Location>, thrust::device_ptr<float> > new_ends = thrust::reduce_by_key(
      dp_locations, dp_locations + num_updates, dp_updates, dp_locs_reduced, dp_lo_reduced);
  num_updates = new_ends.first - dp_locs_reduced;
  printf("\tTook %5.3f to reduce\n", t.GetMs());

  // Copy result from GPU device to host
  t.Start();
  std::vector<Location> location_vector(num_updates);
  std::vector<float> lo_vector(num_updates);
  cudaMemcpy(location_vector.data(), dp_locs_reduced.get(), sizeof(Location) * num_updates, cudaMemcpyDeviceToHost);
  cudaMemcpy(lo_vector.data(), dp_lo_reduced.get(), sizeof(float) * num_updates, cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  printf("\tTook %5.3f to copy to host\n", t.GetMs());

  return OccGrid(location_vector, lo_vector, resolution_);
}

}  // namespace ray_tracing
}  // namespace library
