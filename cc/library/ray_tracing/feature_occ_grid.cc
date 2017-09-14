#include "library/ray_tracing/feature_occ_grid.h"

#include <boost/assert.hpp>

namespace library {
namespace ray_tracing {

const Stats& FeatureOccGrid::GetStats(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(stat_locs_.begin(), stat_locs_.end(), loc);
  BOOST_ASSERT(it != stat_locs_.end() && (*it) == loc);

  size_t pos = it - stat_locs_.begin();
  return stats_[pos];
}

bool FeatureOccGrid::HasStats(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(stat_locs_.begin(), stat_locs_.end(), loc);
  return (it != stat_locs_.end() && (*it) == loc);
}

}  // namespace ray_tracing
}  // namespace library
