#pragma once

#include <vector>
#include <cstddef>

class Histogram {
 public:
  Histogram(double min, double max, double res);

  bool InRange(double val) const;
  double GetMin() const;
  double GetMax() const;

  void Mark(double val);

  int GetCount(double val) const;
  int GetCountsTotal() const;

  double GetProbability(double val) const;
  double GetLikelihood(double val) const;
  double GetCumulativeProbability(double val) const;

  double GetMedian() const;
  double GetPercentile(double percentile) const;

 private:
  double min_;
  double max_;
  double res_;

  double observed_min_ = 0.0;
  double observed_max_ = 0.0;

  std::vector<int> counts_;
  int counts_total_ = 0;

  size_t GetIndex(double val) const;
  double GetValue(size_t idx) const;
};
