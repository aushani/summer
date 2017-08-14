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
  size_t dim = ceil((max - min)/res);
  counts_index_max_ = dim + 2;
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
  if (counts_total_ == 0.0 || val < observed_min_) {
    observed_min_ = val;
  }

  if (counts_total_ == 0.0 || val > observed_max_) {
    observed_max_ = val;
  }

  size_t idx = GetIndex(val);
  counts_[idx] += weight;
  counts_total_+= weight;
}

void Histogram::Clear() {
  counts_.clear();
  counts_total_ = 0.0f;
}

double Histogram::GetCount(double val) const {
  size_t idx = GetIndex(val);

  // Be careful not to increase size of map if we don't have to
  const auto it = counts_.find(idx);
  if (it == counts_.end()) {
    return 0;
  }

  return it->second;
}

double Histogram::GetCountsTotal() const {
  return counts_total_;
}

double Histogram::Sample() const {
  std::uniform_real_distribution<double> rand_cdf(0.0, 1.0);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  double cdf = rand_cdf(rand_engine);
  return GetPercentile(cdf);
}

double Histogram::GetProbability(double val) const {
  if (counts_total_ == 0.0f) {
    return 0.0f;
  }

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
  for (auto it = counts_.cbegin(); it != counts_.cend(); it++) {
    if (GetValue(it->first) > val) {
      break;
    }

    count += it->second;
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

  for (auto it = counts_.cbegin(); it != counts_.cend(); it++) {
    double count_it = it->second;
    double count_left = target_count - count_at;

    if (count_it < count_left) {
      count_at += count_it;
      continue;
    }

    double frac = count_left/count_it;

    double v1 = GetValue(it->first);
    double v2 = GetValue(it->first+1);

    return v1 + (v2 - v1)*frac;
  }

  return max_;
}

double Histogram::GetMedian() const {
  return GetPercentile(0.5);
}

size_t Histogram::GetIndex(double val) const {
  if (val <= min_) {
    return 0;
  }

  if (val >= max_) {
    return counts_index_max_;
  }

  return floor((val - min_) / res_) + 1;
}

double Histogram::GetValue(size_t idx) const {
  if (idx <= 0) {
    return min_;
  }

  if (idx >= counts_index_max_) {
    return max_;
  }

  return (idx-1)*res_ + min_;
}

bool Histogram::IsCompatibleWith(const Histogram &hist) const {
  if (GetMin() != hist.GetMin()) {
    return false;
  }

  if (GetMax() != hist.GetMax()) {
    return false;
  }

  if (GetRes() != hist.GetRes()) {
    return false;
  }

  return true;
}

void Histogram::Add(const Histogram &hist, double weight) {
  BOOST_ASSERT(IsCompatibleWith(hist));

  for (auto it = hist.counts_.cbegin(); it != hist.counts_.cend(); it++) {
    counts_[it->first] += weight * it->second;
  }

  counts_total_ += weight * hist.GetCountsTotal();

  if (hist.observed_min_ < observed_min_) {
    observed_min_ = hist.observed_min_;
  }

  if (hist.observed_max_ > observed_max_) {
    observed_max_ = hist.observed_max_;
  }
}

void Histogram::Blur(double std) {
  double std_range = 3;

  double scale = 1 / sqrt(2 * M_PI * std*std);

  std::map<size_t, double> blurred_counts;

  for (auto it = counts_.begin(); it != counts_.end(); it++) {

    if (it->first == GetIndex(min_)) {
      blurred_counts[it->first] = it->second;
      continue;
    } else if (it->first == GetIndex(max_)) {
      blurred_counts[it->first] = it->second;
      continue;
    }

    double v = GetValue(it->first);
    double v_min = v - std_range * std;
    double v_max = v + std_range * std;

    size_t i0 = GetIndex(v_min);
    size_t i1 = GetIndex(v_max);

    if (i0 <= GetIndex(min_)) {
      i0 = GetIndex(min_) + 1;
    }

    if (i1 >= GetIndex(max_)) {
      i1 = GetIndex(max_) - 1;
    }

    for (size_t i_blur = i0; i_blur <= i1; i_blur++) {
      double v_blur = GetValue(i_blur);
      double d_v = v - v_blur;
      double w = scale*exp(-0.5 * d_v * d_v / (std*std));
      blurred_counts[i_blur] += w * it->second;
    }
  }

  counts_ = blurred_counts;
  counts_total_ = 0;
  for (auto it = counts_.begin(); it != counts_.end(); it ++) {
    counts_total_ += it->second;
  }
}

} // namespace histogram
} // namespace library
