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

  Histogram1GramKey(const ModelObservation &mo, double dist_res, double angle_res,
      double max_size_xy, double max_size_z) {
    idx_theta = std::round( util::MinimizeAngle(mo.theta) / angle_res );
    idx_phi = std::round( util::MinimizeAngle(mo.phi) / angle_res );

    if (std::abs(mo.dist_ray) < max_size_xy) {
      dist_ray_in_range = true;
      idx_dist_ray = std::round(mo.dist_ray / dist_res);
    }

    if (std::abs(mo.dist_z) < max_size_z) {
      dist_z_in_range = true;
      idx_dist_z = std::round(mo.dist_z / dist_res);
    }
  }

  bool InRange() const {
    return dist_ray_in_range && dist_z_in_range;
  }

  bool operator<(const Histogram1GramKey &rhs) const {
    // CHeck to make sure both are in range
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
      int idx_dist_obs = std::round(dist_obs/dist_res);
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

  void MarkObservations(const ObjectState &os, const std::vector<ModelObservation> &obs);

  void PrintStats() const;

  double SampleRange(const ObjectState &os, double sensor_theta, double sensor_phi) const;

 private:
  const double kAngleRes = 0.01;    // ~0.5  degrees
  const double kDistRes = 0.10;     // ~10 cm

  double max_size_xy_ = 5.0;
  double max_size_z_ = 5.0;

  //std::map<HistogramNGramKey, library::histogram::Histogram> histograms_;
  std::map<Histogram1GramKey, library::histogram::Histogram> histograms_;

  void MarkObservation(const ModelObservation &mo);

  bool InRoi(const ModelObservation &mo) const;
  bool IsOccluded(const ModelObservation &mo) const;

  library::histogram::Histogram* GetHistogram(const ModelObservation &mo);
  const library::histogram::Histogram* GetHistogram(const ModelObservation &mo) const;

};

} // namespace kitti
} // namespace app
