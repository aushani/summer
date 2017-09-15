#include "library/ray_tracing/feature_occ_grid.h"

#include <thread>

#include <boost/assert.hpp>
#include <Eigen/Eigenvalues>

namespace library {
namespace ray_tracing {

bool FeatureOccGrid::HasStats(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(stat_locs_.begin(), stat_locs_.end(), loc);
  return (it != stat_locs_.end() && (*it) == loc);
}

const std::vector<Location>& FeatureOccGrid::GetStatLocations() const {
  return stat_locs_;
}

const std::vector<Stats>& FeatureOccGrid::GetStats() const {
  return stats_;
}

const Stats& FeatureOccGrid::GetStats(const Location &loc) const {
  size_t pos = GetStatsPos(loc);
  return stats_[pos];
}

const Eigen::Vector3f& FeatureOccGrid::GetNormal(const Location &loc) const {
  size_t pos = GetStatsPos(loc);
  return normals_[pos];
}

size_t FeatureOccGrid::GetStatsPos(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(stat_locs_.begin(), stat_locs_.end(), loc);
  BOOST_ASSERT(it != stat_locs_.end() && (*it) == loc);

  size_t pos = it - stat_locs_.begin();

  return pos;
}

void FeatureOccGrid::ComputeNormals() {
  int n_threads = 32;

  std::vector<std::thread> threads;
  for (int i=0; i<n_threads; i++) {
    size_t idx0 = i * stat_locs_.size() / n_threads;
    size_t idx1 = (i+1) * stat_locs_.size() / n_threads;
    threads.emplace_back(&FeatureOccGrid::ComputeNormalsWorker, this, idx0, idx1);
  }

  for (auto &t : threads) {
    t.join();
  }
}

void FeatureOccGrid::ComputeNormalsWorker(size_t idx0, size_t idx1) {
  for (size_t idx = idx0; idx < idx1; idx++) {
    ComputeNormalsForIdx(idx);
  }
}

void FeatureOccGrid::ComputeNormalsForIdx(size_t idx) {
  const Location &loc = stat_locs_[idx];

  // Blur
  int blur_size = 1;
  Stats stats;
  for (int di=-blur_size; di<=blur_size; di++) {
    for (int dj=-blur_size; dj<=blur_size; dj++) {
      for (int dk=-blur_size; dk<=blur_size; dk++) {
        Location loc_i(loc.i + di, loc.j + dj, loc.k + dk);

        if (HasStats(loc_i)) {
          stats = stats + GetStats(loc_i);
        }
      }
    }
  }

  normals_[idx] = stats.GetNormal();
  if (normals_[idx].dot(Eigen::Vector3f(loc.i, loc.j, loc.k)) > 0) {
    normals_[idx] *= -1;
  }
}

}  // namespace ray_tracing
}  // namespace library
