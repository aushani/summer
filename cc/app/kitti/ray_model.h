#pragma once

#include <vector>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "library/histogram/histogram.h"
#include "library/util/angle.h"

#include "app/kitti/model_observation.h"

namespace util = library::util;

namespace app {
namespace kitti {

struct Histogram1GramKey {
  int idx_theta=0;
  int idx_phi=0;

  int idx_dist_ray=0;
  bool dist_ray_in_range = false;

  int idx_dist_z=0;
  bool dist_z_in_range = false;

  Histogram1GramKey() {
  }

  Histogram1GramKey(const Histogram1GramKey &key) :
    idx_theta(key.idx_theta),
    idx_phi(key.idx_phi),
    idx_dist_ray(key.idx_dist_ray),
    dist_ray_in_range(key.dist_ray_in_range),
    idx_dist_z(key.idx_dist_z),
    dist_z_in_range(key.dist_z_in_range) {
  }

  Histogram1GramKey(int i_theta, int i_phi, int i_dist_ray, int i_dist_z,
                    double dist_res, double angle_res, double max_size_xy, double max_size_z) :
   idx_theta    (i_theta),
   idx_phi      (i_phi),
   idx_dist_ray (i_dist_ray),
   idx_dist_z   (i_dist_z) {
    int n_angle = std::round(M_PI / angle_res);
    while (idx_theta < -n_angle) idx_theta += 2*n_angle;
    while (idx_theta >  n_angle) idx_theta -= 2*n_angle;

    dist_ray_in_range = (idx_dist_ray * dist_res < max_size_xy);
    dist_z_in_range = (idx_dist_z * dist_res < max_size_z);
  }

  Histogram1GramKey(const ModelObservation &mo, double dist_res, double angle_res, double max_size_xy, double max_size_z) :
    Histogram1GramKey(mo.theta, mo.phi, mo.dist_ray, mo.dist_z, dist_res, angle_res, max_size_xy, max_size_z) {
  }

  Histogram1GramKey(double theta, double phi, double dist_ray, double dist_z,
                    double dist_res, double angle_res, double max_size_xy, double max_size_z) {
    idx_theta = std::floor( util::MinimizeAngle(theta) / angle_res );
    idx_phi = std::floor( util::MinimizeAngle(phi) / angle_res );

    if (std::abs(dist_ray) < max_size_xy) {
      dist_ray_in_range = true;
      idx_dist_ray = std::floor(dist_ray / dist_res);
    }

    if (std::abs(dist_z) < max_size_z) {
      dist_z_in_range = true;
      idx_dist_z = std::floor(dist_z / dist_res);
    }
  }

  bool InRange() const {
    return dist_ray_in_range && dist_z_in_range;
  }

  bool operator<(const Histogram1GramKey &rhs) const {
    // Check to make sure both are in range
    if (!InRange()) {
      return rhs.InRange();
    }

    if (InRange() && !rhs.InRange()) {
      return false;
    }

    if (idx_theta != rhs.idx_theta) {
      return idx_theta < rhs.idx_theta;
    }

    if (idx_phi != rhs.idx_phi) {
      return idx_phi < rhs.idx_phi;
    }

    if (idx_dist_ray != rhs.idx_dist_ray) {
      return idx_dist_ray < rhs.idx_dist_ray;
    }

    return idx_dist_z < rhs.idx_dist_z;
  }

  bool operator==(const Histogram1GramKey &rhs) const {
    return idx_theta    == rhs.idx_theta    &&
           idx_phi      == rhs.idx_phi      &&
           idx_dist_ray == rhs.idx_dist_ray &&
           idx_dist_z   == rhs.idx_dist_z   &&
           InRange()    == rhs.InRange();
  }

  bool operator!=(const Histogram1GramKey &rhs) const {
    return !(*this == rhs);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & idx_theta;
    ar & idx_phi;

    ar & idx_dist_ray;
    ar & dist_ray_in_range;

    ar & idx_dist_z;
    ar & dist_z_in_range;
  }

};

struct HistogramNGramKey {

  std::vector<Histogram1GramKey> idxs_1gram;
  std::vector<int> idxs_dist_obs;

  HistogramNGramKey(const std::vector<ModelObservation> &mos,
      double dist_res, double angle_res, double max_size_xy, double max_size_z) {

    for (const auto& mo : mos) {
      idxs_1gram.emplace_back(mo, dist_res, angle_res, max_size_xy, max_size_z);

      double dist_obs = mo.dist_obs;
      if (dist_obs > max_size_xy) {
        dist_obs = max_size_xy;
      }

      if (dist_obs < -max_size_xy) {
        dist_obs = -max_size_xy;
      }
      int idx_dist_obs = std::floor(dist_obs/dist_res);
      idxs_dist_obs.emplace_back(idx_dist_obs);
    }

    idxs_dist_obs.pop_back();
  }

  bool operator<(const HistogramNGramKey &rhs) const {

    if (idxs_1gram.size() != rhs.idxs_1gram.size()) {
      return idxs_1gram.size() < rhs.idxs_1gram.size();
    }

    for (size_t i = 0; i < idxs_1gram.size(); i++) {
      if (idxs_1gram[i] != rhs.idxs_1gram[i]) {
        return idxs_1gram[i] < rhs.idxs_1gram[i];
      }

      if (i < idxs_dist_obs.size() && idxs_dist_obs[i] != rhs.idxs_dist_obs[i]) {
        return idxs_dist_obs[i] < rhs.idxs_dist_obs[i];
      }
    }

    // Everything equal
    return false;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & idxs_1gram;
    ar & idxs_dist_obs;
  }
};

class RayModel {
 public:
  RayModel();

  void MarkObservations(const std::vector<ModelObservation> &obs);

  std::map<std::pair<double, double>, double> GetHistogramFillinByAngle() const;
  void PrintStats() const;

  void Blur();

  double SampleRange(const ObjectState &os, double sensor_theta, double sensor_phi) const;

  double EvaluateObservations(const std::vector<ModelObservation> &obs) const;

  bool IsRelevant(const ModelObservation &mo) const;

 private:
  double kAngleRes = 0.20;    // ~10 degrees
  double kDistRes = 0.20;     // ~20 cm

  double max_size_xy_ = 5.0;
  double max_size_z_ = 2.5;

  //std::map<HistogramNGramKey, library::histogram::Histogram> histograms_;
  std::map<Histogram1GramKey, library::histogram::Histogram> histograms_;

  void MarkObservation(const ModelObservation &mo);

  bool InRoi(const ModelObservation &mo) const;
  bool IsOccluded(const ModelObservation &mo) const;

  double GetLogLikelihood(const ModelObservation &mo) const;

  library::histogram::Histogram* GetHistogram(const ModelObservation &mo);
  const library::histogram::Histogram* GetHistogram(const ModelObservation &mo) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & kAngleRes;
    ar & kDistRes;

    ar & max_size_xy_;
    ar & max_size_z_;

    ar & histograms_;
  }

};

} // namespace kitti
} // namespace app
