#include "library/ray_tracing/occ_grid.h"

#include <algorithm>

#include <boost/assert.hpp>

namespace library {
namespace ray_tracing {

OccGrid::OccGrid(const std::vector<Location> &locs, const std::vector<float> &los, float res)
    : data_(locs, los, res) {
  BOOST_ASSERT(locs.size() == los.size());
  BOOST_ASSERT(res > 0);
}

OccGrid::OccGrid(const OccGridData &ogd) :
 OccGrid(ogd.locations, ogd.log_odds, ogd.resolution) {
}

float OccGrid::GetLogOdds(Location loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(data_.locations.begin(), data_.locations.end(), loc);
  if (it != data_.locations.end() && (*it) == loc) {
    size_t pos = it - data_.locations.begin();
    return data_.log_odds[pos];
  }

  // Unknown
  return 0.0f;
}

float OccGrid::GetLogOdds(float x, float y, float z) const {
  Location loc(x, y, z, data_.resolution);
  return GetLogOdds(loc);
}

const std::vector<Location>& OccGrid::GetLocations() const {
  return data_.locations;
}

const std::vector<float>& OccGrid::GetLogOdds() const {
  return data_.log_odds;
}

float OccGrid::GetResolution() const {
  return data_.resolution;
}

const OccGridData& OccGrid::GetData() const {
  return data_;
}

}  // namespace ray_tracing
}  // namespace library