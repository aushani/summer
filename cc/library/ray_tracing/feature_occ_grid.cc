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

const Stats& FeatureOccGrid::GetStats(const Location &loc) const {
  size_t pos = GetStatsPos(loc);
  return stats_[pos];
}

const Eigen::Vector3d& FeatureOccGrid::GetNormal(const Location &loc) const {
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
  int n_threads = 4;

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
  Stats stats;
  for (int di=-1; di<=1; di++) {
    for (int dj=-1; dj<=1; dj++) {
      for (int dk=-1; dk<=1; dk++) {
        Location loc_i(loc.i + di, loc.j + dj, loc.k + dk);

        if (HasStats(loc_i)) {
          stats = stats + GetStats(loc_i);
        }
      }
    }
  }

  // Get covariance matrix
  Eigen::Matrix3d cov;
  cov(0, 0) = stats.GetCovX();
  cov(1, 1) = stats.GetCovY();
  cov(2, 2) = stats.GetCovZ();

  cov(0, 1) = stats.GetCovXY();
  cov(1, 0) = stats.GetCovXY();

  cov(0, 2) = stats.GetCovXZ();
  cov(2, 0) = stats.GetCovXZ();

  cov(1, 2) = stats.GetCovYZ();
  cov(2, 1) = stats.GetCovYZ();

  // Solve for eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);

  // Get the normal
  Eigen::Vector3d evals = eig.eigenvalues();
  int idx_min = 0;
  if (std::abs(evals(1)) < std::abs(evals(idx_min))) {
    idx_min = 1;
  }

  if (std::abs(evals(2)) < std::abs(evals(idx_min))) {
    idx_min = 2;
  }

  Eigen::Matrix3d evecs = eig.eigenvectors();
  Eigen::Vector3d normal = evecs.col(idx_min);

  Eigen::Vector3d pos(loc.i, loc.j, loc.k);
  if (pos.dot(normal) > 0) {
    normal *= -1;
  }

  normals_[idx] = normal;
}

}  // namespace ray_tracing
}  // namespace library
