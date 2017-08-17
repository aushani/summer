#pragma once

#include <stdlib.h>
#include <vector>

#include "library/ray_tracing/occ_grid.h"

namespace library {
namespace ray_tracing {

class DenseOccGrid {
 public:
  DenseOccGrid(const OccGrid &og, float max_x, float max_y, float max_z);

  float GetLogOdds(const Location &loc) const;
  float GetProbability(const Location &loc) const;

  float GetResolution() const;

 private:
  static constexpr int kNumThreads = 1;

  const size_t nx_;
  const size_t ny_;
  const size_t nz_;

  const float resolution_;

  std::vector<float> log_odds_;

  bool InRange(const Location &loc) const;
  size_t GetIndex(const Location &loc) const;

  void PopulateDenseWorker(const OccGrid &og, size_t i0, size_t i1);
};

}  // namespace ray_tracing
}  // namespace library
