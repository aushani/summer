// Adapted from dascar
#pragma once

#include <vector>

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

// This struct represents an occupancy grid, holding log odds values for the
// probablity that a given voxel is occupied. Postive means occupied, negative means
// free, and 0 means unknown.
struct OccGrid {
 public:
  // locations and log_odds must be sorted and the same size
  OccGrid(const std::vector<Location> &locs, const std::vector<float> &los, float res);
  OccGrid(const OccGrid &og);
  ~OccGrid();

  float GetLogOdds(Location loc);
  float GetLogOdds(float x, float y, float z);

  // These are parallel containers that need to be sorted and the same size
  const std::vector<Location> locations;
  const std::vector<float> log_odds;

  const float resolution;
};

}  // namespace ray_tracing
}  // namespace library
