#pragma once

#include <vector>
#include <cstddef>

class Histogram {
 public:
  Histogram(double min, double max, double res);

  bool InRange(double val) const;

  void Mark(double val);

  int GetCount(double val) const;
  int GetCountsTotal() const;

  double GetProbability(double val) const;
  double GetCumulativeProbability(double val) const;

 private:
  double min_;
  double max_;
  double res_;

  std::vector<int> counts_;
  int counts_total_ = 0;

  size_t GetIndex(double val) const;
};
