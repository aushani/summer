#include "library/detection_mapping/builder.h"

#include <iostream>

#include <boost/assert.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include "library/timer/timer.h"

namespace tr = library::timer;

namespace library {
namespace detection_mapping {

// This is the data we need on the device
struct DeviceData {
  DeviceData(float range_x, float range_y, float resolution, int max_observations, float max_range);

  void FreeDeviceMemory();

  void CopyData(const std::vector<Eigen::Vector3d> &hits);

  // Centered on (0, 0)
  __host__ __device__ int GetResultIndex(int i, int j) const;

  int num_observations = 0;
  int max_voxel_visits_per_ray = 0;

  float resolution = 0.0f;

  float *hit_x = nullptr;
  float *hit_y = nullptr;
  float *hit_z = nullptr;

  float *result = nullptr;
  int n_x = 0;
  int n_y = 0;
};

DeviceData::DeviceData(float range_x, float range_y, float resolution, int max_observations, float max_range)
    : resolution(resolution), max_voxel_visits_per_ray(max_range / resolution) {
  // Allocate memory on the device.
  cudaError_t err = cudaMalloc(&hit_x, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_y, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  err = cudaMalloc(&hit_z, sizeof(float) * max_observations);
  BOOST_ASSERT(err == cudaSuccess);

  int n_x = 2*ceil(range_x / resolution) + 1;
  int n_y = 2*ceil(range_y / resolution) + 1;
  err = cudaMalloc(&result, sizeof(float) * n_x * n_y);
  BOOST_ASSERT(err == cudaSuccess);
}

void DeviceData::FreeDeviceMemory() {
  cudaFree(hit_x);
  cudaFree(hit_y);
  cudaFree(hit_z);

  cudaFree(result);
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

__host__ __device__ int DeviceData::GetResultIndex(int i, int j) const {
  int di = i - n_x/2;
  int dj = j - n_y/2;

  if (di < 0 || di >= n_x ||
      dj < 0 || dj >= n_y) {
    return -1;
  }

  return di * n_y + dj;
}

Builder::Builder(double range_x, double range_y, float resolution, int max_observations, float max_range)
    : max_observations_(max_observations),
      resolution_(resolution),
      device_data_(new DeviceData(range_x, range_y, resolution, max_observations, max_range)) {
}

Builder::~Builder() {
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

  // TODO Get origin and hit relative to pose
  float hx = data.hit_x[hit_idx];
  float hy = data.hit_y[hit_idx];
  float hz = data.hit_z[hit_idx];

  float hit[3] = {hx, hy, hz};
  float origin[3] = {0, 0, 0};

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
  bool valid = true;
  for (int step = 0; step < (data.max_voxel_visits_per_ray - 1); ++step) {
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
    }

    // Now write out to result
    for (int di=-5; di<=5; di++) {
      for (int dj=-5; dj<=5; dj++) {
        int res_idx = data.GetResultIndex(cur_loc[0], cur_loc[1]);
        if (res_idx >= 0) {
          data.result[res_idx]++;
        }
      }
    }
  }

  // Done ray tracing, write out endpoint
  for (int di=-5; di<=5; di++) {
    for (int dj=-5; dj<=5; dj++) {
      int res_idx = data.GetResultIndex(cur_loc[0], cur_loc[1]);
      if (res_idx >= 0) {
        data.result[res_idx]--;
      }
    }
  }
}

void Builder::GenerateDetectionMap(const std::vector<Eigen::Vector3d> &hits) {
  library::timer::Timer t;

  BOOST_ASSERT(hits.size() <= max_observations_);

  // Check for empty data
  if (hits.size() == 0) {
    return;
  }

  // First, we need to send the data to the GPU device
  //t.Start();
  device_data_->CopyData(hits);
  //printf("\tTook %5.3f ms to copy data\n", t.GetMs());

  // Now run ray tracing on the GPU device
  t.Start();
  int blocks = ceil(static_cast<float>(device_data_->num_observations) / kThreadsPerBlock_);
  RayTracingKernel<<<blocks, kThreadsPerBlock_>>>(*device_data_);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
  printf("\tTook %5.3f ms to run kernel\n", t.GetMs());

  return;
}

}  // namespace ray_tracing
}  // namespace library
