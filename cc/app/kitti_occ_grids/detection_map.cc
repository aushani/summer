#include "app/kitti_occ_grids/detection_map.h"

#include <cmath>

#include <boost/assert.hpp>

namespace app {
namespace kitti_occ_grids {

DetectionMap::DetectionMap(double range_x, double range_y, double resolution, const std::vector<std::string> &classes) :
 resolution_(resolution),
 n_classes_(classes.size()),
 n_x_(2*std::ceil(range_x / resolution) + 1),
 n_y_(2*std::ceil(range_y / resolution) + 1),
 scores_(n_x_ * n_y_ * n_classes_) {
  printf("+- %5.3f, %5.3f m at %5.3f res = %ld, %ld\n", range_x, range_y, resolution_, n_x_, n_y_);
  for (const auto &cn : classes) {
    classes_[cn] = classes_.size();
  }
}

void DetectionMap::Update(int i, int j, const std::string &classname, double update) {
  int idx = GetIndex(i, j, classname);
  if (idx >= 0) {
    scores_[idx] += update;
  }
}

double DetectionMap::GetScore(int i, int j, const std::string &classname) const {
  int idx = GetIndex(i, j, classname);
  if (idx >= 0) {
    return scores_[idx];
  } else {
    return 0.0;
  }
}

double DetectionMap::GetProbability(int i, int j, const std::string &classname) const {
  if (GetIndex(i, j, classname) < 0) {
    return 0.0;
  }

  double my_score = GetScore(i, j, classname);

  double max_score = my_score;
  for (const auto &kv : classes_) {
    double score = GetScore(i, j, kv.first);
    if (score > max_score) {
      max_score = score;
    }
  }

  double my_weight = exp(my_score - max_score);

  double total_weight = 0.0;
  for (const auto &kv : classes_) {
    double score = GetScore(i, j, kv.first);
    total_weight += exp(score - max_score);
  }

  return my_weight / total_weight;
}

std::vector<std::string> DetectionMap::GetClasses() const {
  std::vector<std::string> classes;

  for (const auto &kv : classes_) {
    classes.push_back(kv.first);
  }

  return classes;
}

size_t DetectionMap::GetNX() const {
  return n_x_;
}

size_t DetectionMap::GetNY() const {
  return n_y_;
}

double DetectionMap::GetResolution() const {
  return resolution_;
}

int DetectionMap::GetIndex(int i, int j, const std::string &classname) const {
  int is = i + n_x_/2;
  int js = j + n_y_/2;

  if (is >= n_x_ || is < 0 ||
      js >= n_y_ || js < 0) {
    return -1;
  }

  auto it_cs = classes_.find(classname);
  if (it_cs == classes_.end()) {
    return -1;
  }

  int cs = it_cs->second;

  return ((is * n_y_) + js ) * n_classes_ + cs;
}

} // namespace kitti_occ_grids
} // namespace app
