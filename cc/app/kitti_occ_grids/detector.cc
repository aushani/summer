#include "app/kitti_occ_grids/detector.h"

#include <stdlib.h>
#include <iostream>
#include <thread>

#include <boost/assert.hpp>

namespace app {
namespace kitti_occ_grids {

Detector::Detector(double res, double range_x, double range_y) :
 range_x_(range_x), range_y_(range_y),
  n_x_(2 * std::ceil(range_x / res) + 1),
  n_y_(2 * std::ceil(range_y / res) + 1),
  res_(res) {
}

void Detector::AddModel(const std::string &classname, const clt::MarginalModel &mm) {
  classnames_.push_back(classname);
  detector_.AddModel(mm);
}

void Detector::Evaluate(const rt::DeviceOccGrid &scene) {
  detector_.Run(scene);
}

double Detector::GetScore(const std::string &classname, const ObjectState &os) const {
  if (!InRange(os)) {
    return 0.0;
  }

  const auto &it = class_scores_.find(classname);
  BOOST_ASSERT(it != class_scores_.end());
  const auto &scores = it->second;

  size_t idx = GetIndex(os);
  return scores[idx];
}

double Detector::GetProb(const std::string &classname, const ObjectState &os) const {
  if (!InRange(os)) {
    return 0.0;
  }

  size_t idx = GetIndex(os);

  const auto &it = class_scores_.find(classname);
  BOOST_ASSERT(it != class_scores_.end());
  const auto &scores = it->second;
  double my_score = scores[idx];

  double max_score = my_score;

  for (const auto &kv : class_scores_) {
    const auto &classname = kv.first;

    const auto &it = class_scores_.find(classname);
    BOOST_ASSERT(it != class_scores_.end());
    const auto &scores = it->second;
    double score = scores[idx];

    if (score > max_score) {
      max_score = score;
    }
  }

  double sum = 0;

  for (const auto &kv : class_scores_) {
    const auto &classname = kv.first;

    const auto &it = class_scores_.find(classname);
    BOOST_ASSERT(it != class_scores_.end());
    const auto &scores = it->second;
    double score = scores[idx];

    sum += exp(score - max_score);
  }

  double prob = exp(my_score - max_score) / sum;
  double lo = -log(1/prob - 1);
  if (lo > 10) {
    lo = 10;
  }

  if (lo < -10) {
    lo = -10;
  }

  return lo;
}

double Detector::GetRangeX() const {
  return range_x_;
}

double Detector::GetRangeY() const {
  return range_y_;
}

bool Detector::InRange(const ObjectState &os) const {
  return std::abs(os.x) < range_x_ && std::abs(os.y) < range_y_;
}

size_t Detector::GetIndex(const ObjectState &os) const {
  int ix = os.x / res_ + n_x_ / 2;
  int iy = os.y / res_ + n_y_ / 2;

  if (ix >= n_x_ || iy >= n_y_) {
    return -1;
  }

  size_t idx = ix * n_y_ + iy;
  return idx;
}

ObjectState Detector::GetState(size_t idx) const {
  size_t ix = idx / n_y_;
  size_t iy = idx % n_y_;

  // int instead of size_t because could be negative
  int dix = ix - n_x_/2;
  int diy = iy - n_y_/2;

  double x = dix * res_;
  double y = diy * res_;

  return ObjectState(x, y, 0);
}

double Detector::GetRes() const {
  return res_;
}

} // namespace kitti_occ_grids
} // namespace app
