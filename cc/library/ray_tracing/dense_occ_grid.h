#pragma once

#include <stdlib.h>
#include <vector>

#include "library/ray_tracing/occ_grid.h"

namespace library {
namespace ray_tracing {

class DenseOccGrid {
 public:
  DenseOccGrid(const OccGrid &og, float max_x, float max_y, float max_z, bool make_binary);

  void Set(const Location &loc, float p);
  void Clear(const Location &loc);

  float GetProbability(const Location &loc) const;
  bool IsKnown(const Location &loc) const;
  double FractionKnown() const;
  size_t Size() const;

  bool InRange(const Location &loc) const;
  float GetResolution() const;

 private:
  static constexpr int kNumThreads = 1;

  const int nx_;
  const int ny_;
  const int nz_;

  const float resolution_;

  std::vector<float> probs_;
  std::vector<bool> known_;

  size_t GetIndex(const Location &loc) const;

  void PopulateDenseWorker(const OccGrid &og, size_t i0, size_t i1, bool make_binary);
};

}  // namespace ray_tracing
}  // namespace library
