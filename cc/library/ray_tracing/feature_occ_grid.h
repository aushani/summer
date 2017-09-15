#pragma once

#include <vector>

#include <Eigen/Core>

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/stats.h"

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

class FeatureOccGrid : public OccGrid {
 public:
  FeatureOccGrid(const std::vector<Location> &locs, const std::vector<float> &los,
                 const std::vector<Location> &stat_locs, const std::vector<Stats> &stats, float res) :
     OccGrid(locs, los, res), stat_locs_(stat_locs), stats_(stats), normals_(stat_locs.size()) { }

  bool HasStats(const Location &loc) const;

  const Stats& GetStats(const Location &loc) const;
  const Eigen::Vector3d& GetNormal(const Location &loc) const;

  void ComputeNormals();

 private:
  // Parallel containers
  const std::vector<Location> stat_locs_;
  const std::vector<Stats> stats_;

  std::vector<Eigen::Vector3d> normals_;

  size_t GetStatsPos(const Location &loc) const;

  void ComputeNormalsForIdx(size_t idx);
  void ComputeNormalsWorker(size_t idx0, size_t idx1);
};

}  // namespace ray_tracing
}  // namespace library
