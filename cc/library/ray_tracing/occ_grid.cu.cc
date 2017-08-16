#include "library/ray_tracing/occ_grid.h"

#include <algorithm>

#include <boost/assert.hpp>

namespace library {
namespace ray_tracing {

OccGrid::OccGrid(const std::vector<Location> &locs, const std::vector<float> &los, float res)
    : locations(locs), log_odds(los), resolution(res) {
  BOOST_ASSERT(locations.size() == log_odds.size());
  BOOST_ASSERT(resolution > 0);
}

OccGrid::OccGrid(const OccGrid &og) : locations(og.locations), log_odds(og.log_odds),
  resolution(og.resolution) {}

OccGrid::~OccGrid() {}

float OccGrid::GetLogOdds(Location loc) {
  std::vector<Location>::const_iterator it = std::lower_bound(locations.begin(), locations.end(), loc);
  if (it != locations.end() && (*it) == loc) {
    size_t pos = it - locations.begin();
    return log_odds[pos];
  }

  // Unknown
  return 0.0f;
}

float OccGrid::GetLogOdds(float x, float y, float z) {
  Location loc(x, y, z, resolution);
  return GetLogOdds(loc);
}

}  // namespace ray_tracing
}  // namespace library
