#pragma once

#include <vector>

namespace library {
namespace feature {

struct Counter {
  int counts_occu[2] = {0, 0};

  std::vector<int> counts_features;
  int total_feature_counts = 0;

  float angle_res = 0;

  int n_phi = 0;
  int n_theta = 0;

  Counter() : Counter(2*M_PI) {}

  Counter(float ar) :
   angle_res(ar),
   n_phi((2*M_PI / angle_res) - 1),      // after we get too high, just call everything a vertical vector
   n_theta( ceil(2*M_PI / angle_res)) {
    int n_angle_bins = n_phi * n_theta + 1;
    counts_features.resize(n_angle_bins, 0);
  }

  void Count(bool occu) {
    counts_occu[GetIndex(occu)]++;
  }

  void Count(float theta, float phi) {
    // Make sure phi is always positive
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

  int GetMode(float *theta, float *phi) const {
    int max_i = 0;
    int max_count = counts_features[0];
    for (size_t i = 1; i<counts_features.size(); i++) {
      if (counts_features[i] > max_count) {
        max_i = i;
        max_count = counts_features[i];
      }
    }

    GetAngles(max_i, theta, phi);

    return max_count;
  }

  int GetIndex(bool occu) const {
    return occu ? 0 : 1;
  }

  int GetIndex(float theta, float phi) const {
    while (theta < 0)      theta += 2*M_PI;
    while (theta > 2*M_PI) theta -= 2*M_PI;

    while (phi < 0)      phi += 2*M_PI;
    while (phi > 2*M_PI) phi -= 2*M_PI;

    int n_t = theta / angle_res;
    int n_p = phi / angle_res;

    // Check for vertical
    if (n_p >= n_phi) {
      return n_phi * n_theta;
    }

    BOOST_ASSERT(n_t < n_theta);

    int idx =  n_t * n_phi + n_p;

    return idx;
  }

  void GetAngles(int idx, float *theta, float *phi) const {
    if (idx == n_theta * n_phi) {
      (*theta) = 0;
      (*phi) = M_PI/2;
    } else {
      int n_t = idx / n_phi;
      int n_p = idx % n_phi;

      (*theta) = n_t * angle_res;
      (*phi) = n_p * angle_res;
    }
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
    ar & n_theta;
    ar & n_phi;
  }
};

} // namespace chow_liu_tree
} // namespace library
