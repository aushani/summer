#include "histogram.h"

#include <math.h>

Histogram::Histogram(double min, double max, double res) :
 min_(min), max_(max), res_(res), counts_total_(0) {
  int dim = ceil((max - min)/res);
  counts_ = std::vector<int>(dim + 3, 0); // Add 2 so we keep track of values out of range as well, plus 1 for rounding
}

bool Histogram::InRange(double val) const {
  return val >= min_ && val <= max_;
}

void Histogram::Mark(double val) {
  size_t idx = GetIndex(val);
  counts_[idx]++;
  counts_total_++;
}

int Histogram::GetCount(double val) const {
  size_t idx = GetIndex(val);
  return counts_[idx];
}

int Histogram::GetCountsTotal() const {
  return counts_total_;
}

double Histogram::GetProbability(double val) const {
  if (counts_total_ == 0)
    return 0;

  int count = GetCount(val);
  return (count + 0.0f)/counts_total_;
}

double Histogram::GetCumulativeProbability(double val) const {
  int count = 0;
  for (size_t i=0; i<=GetIndex(val); i++) {
    count += counts_[i];
  }
  return (count + 0.0f)/counts_total_;
}

size_t Histogram::GetIndex(double val) const {
  if (val <= min_)
    return 0;
  if (val >= max_)
    return counts_.size() - 1;

  return round((val - min_) / res_) + 1;
}
