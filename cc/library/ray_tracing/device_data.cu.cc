#include "library/ray_tracing/device_data.cu.h"

#include <boost/assert.hpp>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "library/timer/timer.h"

namespace library {
namespace ray_tracing {

DeviceData::DeviceData(float resolution, float max_range, int max_observations)
    : resolution(resolution), max_voxel_visits_per_ray(max_range / resolution) {
  // Allocate memory on the device.
  size_t alloced = 0;
  cudaError_t err = cudaMalloc(&hit_x, sizeof(float) * max_observations);
  alloced += sizeof(float) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_y, sizeof(float) * max_observations);
  alloced += sizeof(float) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_z, sizeof(float) * max_observations);
  alloced += sizeof(float) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_intensity, sizeof(float) * max_observations);
  alloced += sizeof(float) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&locations, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  alloced += sizeof(Location) * max_observations * max_voxel_visits_per_ray;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  alloced += sizeof(float) * max_observations * max_voxel_visits_per_ray;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&stats_locs, sizeof(Location) * max_observations);
  alloced += sizeof(Location) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&stats, sizeof(Stats) * max_observations);
  alloced += sizeof(Stats) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&locations_reduced, sizeof(Location) * max_observations * max_voxel_visits_per_ray);
  alloced += sizeof(Location) * max_observations * max_voxel_visits_per_ray;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&log_odds_updates_reduced, sizeof(float) * max_observations * max_voxel_visits_per_ray);
  alloced += sizeof(float) * max_observations * max_voxel_visits_per_ray;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&stats_locs_reduced, sizeof(Location) * max_observations);
  alloced += sizeof(Location) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&stats_reduced, sizeof(Stats) * max_observations);
  alloced += sizeof(Stats) * max_observations;
  BOOST_ASSERT(err == cudaSuccess);

  printf("Alloced %ld MBytes for OccGridBuilder\n", alloced/(1024*1024));
}

void DeviceData::FreeDeviceMemory() {
  cudaFree(hit_x);
  cudaFree(hit_y);
  cudaFree(hit_z);
  cudaFree(hit_intensity);
  cudaFree(locations);
  cudaFree(log_odds_updates);
  cudaFree(stats_locs);
  cudaFree(stats);
  cudaFree(locations_reduced);
  cudaFree(log_odds_updates_reduced);
  cudaFree(stats_locs_reduced);
  cudaFree(stats_reduced);
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

void DeviceData::CopyIntensity(const std::vector<float> &intensity) {
  size_t sz_copy = sizeof(float) * intensity.size();

  cudaMemcpy(hit_intensity, intensity.data(), sz_copy, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
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

  // Stats!
  if (data.compute_stats) {
    Stats stats;
    stats.count = 1;

    stats.intensity = data.hit_intensity[hit_idx];

    stats.x = hit[0];
    stats.y = hit[1];
    stats.z = hit[2];

    data.stats_locs[hit_idx] = loc;
    data.stats[hit_idx] = stats;
  } else {
    Stats stats;
    stats.intensity = 0;
    stats.count = 0;

    data.stats_locs[hit_idx] = loc;
    data.stats[hit_idx] = stats;
  }
}

struct NoUpdate {
  __host__ __device__ bool operator()(const float x) const { return fabs(x) < 1e-6; }
};

void DeviceData::RunKernel(bool cs) {
  //library::timer::Timer t;

  compute_stats = cs;

  //t.Start();
  int blocks = ceil(static_cast<float>(num_observations) / kThreadsPerBlock);
  RayTracingKernel<<<blocks, kThreadsPerBlock>>>(*this);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  //printf("\tTook %5.3f ms to run kernel\n", t.GetMs());
}

size_t DeviceData::ReduceLogOdds() {
  boost::optional<OutOfRange> oor;
  return ReduceLogOdds(oor);
}

size_t DeviceData::ReduceLogOdds(const boost::optional<OutOfRange> &oor) {
  // Accumulate all the updates
  size_t num_updates = num_observations * max_voxel_visits_per_ray;

  // First prune unnecessary updates
  //t.Start();
  thrust::device_ptr<Location> dp_locations(locations);
  thrust::device_ptr<float> dp_updates(log_odds_updates);

  auto dp_locations_end = thrust::remove_if(dp_locations, dp_locations + num_updates, dp_updates, NoUpdate());
  auto dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, NoUpdate());
  //printf("\tTook %5.3f to prune from %ld to %ld\n", t.GetMs(), num_updates, dp_locations_end - dp_locations);
  num_updates = dp_locations_end - dp_locations;

  // Prune updates that are out of range
  if (oor) {
    //t.Start();
    auto dp_updates_end = thrust::remove_if(dp_updates, dp_updates + num_updates, dp_locations, *oor);
    auto dp_locations_end = thrust::remove_if(dp_locations, dp_locations + num_updates, *oor);
    //printf("\tTook %5.3f to enfore in range (%ld->%ld)\n",
    //       t.GetMs(), num_updates, dp_locations_end - dp_locations);
    num_updates = dp_locations_end - dp_locations;
  }

  // Now reduce updates to resulting log odds
  //t.Start();
  thrust::sort_by_key(dp_locations, dp_locations + num_updates, dp_updates);
  //printf("\tTook %5.3f to sort\n", t.GetMs());

  //t.Start();
  thrust::device_ptr<Location> dp_locs_reduced(locations_reduced);
  thrust::device_ptr<float> dp_lo_reduced(log_odds_updates_reduced);

  thrust::pair<thrust::device_ptr<Location>, thrust::device_ptr<float> > new_ends = thrust::reduce_by_key(
      dp_locations, dp_locations + num_updates, dp_updates, dp_locs_reduced, dp_lo_reduced);
  num_updates = new_ends.first - dp_locs_reduced;
  //printf("\tTook %5.3f to reduce\n", t.GetMs());

  return num_updates;
}

size_t DeviceData::ReduceStats() {
  size_t num_stats = num_observations;

  thrust::device_ptr<Location> dp_locations(stats_locs);
  thrust::device_ptr<Stats> dp_stats(stats);

  thrust::sort_by_key(dp_locations, dp_locations + num_stats, dp_stats);

  thrust::device_ptr<Location> dp_locs_reduced(stats_locs_reduced);
  thrust::device_ptr<Stats> dp_stats_reduced(stats_reduced);

  auto new_ends = thrust::reduce_by_key(dp_locations, dp_locations + num_stats, dp_stats, dp_locs_reduced, dp_stats_reduced);
  num_stats = new_ends.first - dp_locs_reduced;

  return num_stats;
}

}  // namespace ray_tracing
}  // namespace library
