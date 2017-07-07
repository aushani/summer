#pragma once

#include <vector>
#include <stdio.h>

#include "library/geometry/point.h"

namespace ge = library::geometry;

class ObjectModel {
 public:
  ObjectModel(double size, double res);

  bool InBounds(const ge::Point &x) const;

  double EvaluateLikelihood(const ge::Point &x, double label) const;
  void Build(const ge::Point &x, double label);

 private:
  double size_;
  double res_;
  size_t dim_size_;

  std::vector<int> counts_;
  int sum_counts_ = 0;

  int LookupCount(const ge::Point &x, double label) const;
  int GetTableIndex(const ge::Point &x, double label) const;
};
