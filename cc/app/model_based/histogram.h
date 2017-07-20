#pragma once

#include <vector>
#include <cstddef>

class Histogram {
 public:
  Histogram(double min, double max, double res);

  bool InRange(double val) const;
  double GetMin() const;
  double GetMax() const;
  double GetRes() const;

  void Mark(double val, double weight=1.0);
  void Clear();

  double GetCount(double val) const;
  double GetCountsTotal() const;

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

  std::vector<double> counts_;
  double counts_total_ = 0.0f;

  size_t GetIndex(double val) const;
  double GetValue(size_t idx) const;
};
