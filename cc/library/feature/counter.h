#pragma once

#include <vector>

namespace library {
namespace feature {

struct Counter {
  int counts_occu[2] = {0, 0};

  std::vector<int> counts_features;
  int total_feature_counts = 0;

  float angle_res = 0;
  int n_x = 0;

  Counter() : Counter(2*M_PI) {}

  Counter(float ar) : angle_res(ar), n_x(std::ceil(2*M_PI / angle_res)) {
    int n_angle_bins = n_x * n_x;
    counts_features.resize(n_angle_bins, 0);
  }

  void Count(bool occu) {
    counts_occu[GetIndex(occu)]++;
  }

  void Count(float theta, float phi) {
    if (phi < 0) {
      phi *= -1;
      theta = theta + M_PI;
    }

    counts_features[GetIndex(theta, phi)]++;
    total_feature_counts++;
  }

  int GetCount(bool occu) const {
    return counts_occu[GetIndex(occu)];
  }

  int GetCount(float theta, float phi) const {
    return counts_features[GetIndex(theta, phi)];
  }

  int GetIndex(bool occu) const {
    return occu ? 0 : 1;
  }

  int GetIndex(float theta, float phi) const {
    while (theta < 0)      theta += 2*M_PI;
    while (theta > 2*M_PI) theta -= 2*M_PI;

    while (phi < 0)      phi += 2*M_PI;
    while (phi > 2*M_PI) phi -= 2*M_PI;

    int n_t = round(theta / angle_res);
    int n_p = round(phi / angle_res);

    return n_t * n_x + n_p;
  }

  int GetNumOccuObservations() const {
    return counts_occu[0] + counts_occu[1];
  }

  int GetNumFeatureObservations() const {
    return total_feature_counts;
  }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & counts_occu;

    ar & counts_features;
    ar & total_feature_counts;

    ar & angle_res;
    ar & n_x;
  }
};

} // namespace chow_liu_tree
} // namespace library
