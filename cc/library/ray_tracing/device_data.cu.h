#pragma once

#include <vector>
#include <Eigen/Core>

#include "library/ray_tracing/stats.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

struct OutOfRange {
  size_t max_i = 0;
  size_t max_j = 0;
  size_t max_k = 0;

  OutOfRange() {}

  OutOfRange(size_t mi, size_t mj, size_t mk) :
   max_i(mi), max_j(mj), max_k(mk) {
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


// This is the data we need to generate occupancy grids that
// is passed to the device. Package it in this way so we can
// take advantage of coalesced memory operations on the GPU
// for faster performance.
struct DeviceData {
  DeviceData(float resolution, float max_range, int max_observations);

  void FreeDeviceMemory();

  void CopyData(const std::vector<Eigen::Vector3d> &hits);
  void CopyIntensity(const std::vector<float> &intensity);

  void RunKernel(bool compute_stats);
  size_t ReduceLogOdds();

  size_t ReduceStats();

  int num_observations = 0;
  int max_voxel_visits_per_ray = 0;

  float resolution = 0.0f;

  bool compute_stats = false;

  float *hit_x = nullptr;
  float *hit_y = nullptr;
  float *hit_z = nullptr;
  float *hit_intensity = nullptr;

  OutOfRange oor;
  bool oor_valid = false;
  int steps_per_ray = 0;

  Location *locations = nullptr;
  float *log_odds_updates = nullptr;

  Location *stats_locs = nullptr;
  Stats *stats = nullptr;

  Location *locations_reduced = nullptr;
  float *log_odds_updates_reduced = nullptr;

  Location *stats_locs_reduced = nullptr;
  Stats *stats_reduced = nullptr;

  float pose_xyz[3] = {0.0f, 0.0f, 0.0f};
  float pose_theta = 0.0f;

  static constexpr float kLogOddsFree = -0.1;
  static constexpr float kLogOddsOccupied = 1.0;
  static constexpr float kLogOddsUnknown = 0.0;

  static constexpr int kThreadsPerBlock = 1024;
};

}  // namespace ray_tracing
}  // namespace library
