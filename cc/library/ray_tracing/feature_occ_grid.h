#pragma once

#include <vector>

#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/stats.h"

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

class FeatureOccGrid : public OccGrid {
 public:
  FeatureOccGrid(const std::vector<Location> &locs, const std::vector<float> &los,
                 const std::vector<Location> &stat_locs, const std::vector<Stats> &stats, float res) :
 OccGrid(locs, los, res), stat_locs_(stat_locs), stats_(stats) { }

  const Stats& GetStats(const Location &loc) const;
  bool HasStats(const Location &loc) const;

 private:
  // Parallel containers
  const std::vector<Location> stat_locs_;
  const std::vector<Stats> stats_;

};

}  // namespace ray_tracing
}  // namespace library
