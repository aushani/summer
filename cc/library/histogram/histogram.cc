#include "library/histogram/histogram.h"

#include <math.h>
#include <random>
#include <chrono>

namespace library {
namespace histogram {

Histogram::Histogram() {

}

Histogram::Histogram(double min, double max, double res) :
 min_(min), max_(max), res_(res), counts_total_(0.0f) {
  int dim = ceil((max - min)/res);
  counts_ = std::vector<double>(dim + 3, 0.0f); // Add 2 so we keep track of values out of range as well, plus 1 for rounding
}

bool Histogram::InRange(double val) const {
  return val >= min_ && val <= max_;
}

double Histogram::GetMin() const {
  return min_;
}

double Histogram::GetMax() const {
  return max_;
}

double Histogram::GetRes() const {
  return res_;
}

void Histogram::Mark(double val, double weight) {
  if (counts_total_ == 0.0 || val < observed_min_)
    observed_min_ = val;

  if (counts_total_ == 0.0 || val > observed_max_)
    observed_max_ = val;

  size_t idx = GetIndex(val);
  counts_[idx] += weight;
  counts_total_+= weight;
}

void Histogram::Clear() {
  for (size_t i = 0; i<counts_.size(); i++) {
    counts_[i] = 0.0f;
  }
  counts_total_ = 0.0f;
}

double Histogram::GetCount(double val) const {
  size_t idx = GetIndex(val);
  return counts_[idx];
}

double Histogram::GetCountsTotal() const {
  return counts_total_;
}

double Histogram::Sample() const {
  std::uniform_real_distribution<double> rand_cdf(0.0, 1.0);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  double cdf = rand_cdf(rand_engine);

  double cdf_at = 0;
  size_t idx_at = 0;

  double v1 = observed_min_;
  double cdf_left = 0;

  while (cdf_at < cdf && idx_at < counts_.size()) {
    v1 = GetValue(idx_at);
    cdf_left = (cdf - cdf_at);

    idx_at++;
    cdf_at += counts_[idx_at]/counts_total_;
  }

  // Linearly interpolate?
  double v2 = observed_max_;
  if (idx_at < counts_.size())
    v2 = GetValue(idx_at);

  double p = counts_[idx_at]/counts_total_;

  double res = v1 + (v2-v1)*(cdf_left/p);

  return res;
}

double Histogram::GetProbability(double val) const {
  if (counts_total_ == 0.0f)
    return 0.0f;

  double count = GetCount(val);
  return count/counts_total_;
}

double Histogram::GetLikelihood(double val) const {
  if (val <= min_) {
    double denom = min_ - observed_min_;
    if (denom < res_) {
      denom = res_;
    }
    return GetProbability(val)/denom;
  }

  if (val >= max_) {
    double denom = min_ - observed_min_;
    if (denom < res_) {
      denom = res_;
    }
    return GetProbability(val)/denom;
  }

  return GetProbability(val)/res_;
}

double Histogram::GetCumulativeProbability(double val) const {
  int count = 0;
  for (size_t i=0; i<=GetIndex(val); i++) {
    count += counts_[i];
  }
  return count/counts_total_;
}

double Histogram::GetPercentile(double percentile) const {
  if (percentile <= 0) {
    return min_;
  }

  if (percentile >= 1) {
    return max_;
  }

  double count_at = 0;
  double target_count = percentile * counts_total_;

  for (size_t i=0; i < counts_.size(); i++) {
    double count_i = counts_[i];
    double count_left = target_count - count_at;

    if (count_i < count_left) {
      count_at += count_i;
      continue;
    }

    double frac = count_left/count_i;

    double v1 = GetValue(i);
    double v2 = GetValue(i+1);

    return v1 + (v2 - v1)*frac;
  }

  return max_;
}

double Histogram::GetMedian() const {
  return GetPercentile(0.5);
}

size_t Histogram::GetIndex(double val) const {
  if (val <= min_)
    return 0;
  if (val >= max_)
    return counts_.size() - 1;

  return floor((val - min_) / res_) + 1;
}

double Histogram::GetValue(size_t idx) const {
  if (idx <= 0)
    return min_;
  if (idx >= counts_.size()-1)
    return max_;

  return (idx-1)*res_ + min_;
}

} // namespace histogram
} // namespace library
