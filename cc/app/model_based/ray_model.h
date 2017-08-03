#pragma once

#include <vector>
#include <stdio.h>
#include <cstddef>

#include <Eigen/Core>

#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "histogram.h"
#include "observation.h"
#include "object_state.h"

struct ModelObservation {
  double phi;       // Angle of relative relative to object
  double dist_ray;  // Closest distance of ray to object center
  double dist_obs;  // Distance of range observation relative to closest point of ray to object center

  bool in_front;    // Check to make sure that the object is not behind us

  ModelObservation(const ObjectState &os, const Observation &x_hit) {
    // Compute relative angle with object
    phi = x_hit.GetTheta() - os.GetTheta();
    while (phi < 0)      phi += 2*M_PI;
    while (phi > 2*M_PI) phi -= 2*M_PI;

    // Compute ray's distance from object center
    dist_ray = x_hit.GetSinTheta()*os.GetPos()(0) - x_hit.GetCosTheta()*os.GetPos()(1);

    // Compute location of ray hit relative to object
    dist_obs = x_hit.GetRange() - x_hit.GetCosTheta()*os.GetPos()(0) - x_hit.GetSinTheta()*os.GetPos()(1);

    in_front = dist_obs < x_hit.GetRange();
  }
};

struct Histogram1GramKey {
  int idx_phi;
  int idx_dist;

  Histogram1GramKey() :
    idx_phi(0), idx_dist(0) {
  }

  Histogram1GramKey(const ModelObservation &mo, double phi_step, double distance_step) :
    Histogram1GramKey(mo.phi, mo.dist_ray, phi_step, distance_step) {
  }

  Histogram1GramKey(double phi, double dist_ray, double phi_step, double distance_step) {
    while (phi < -M_PI)      phi += 2*M_PI;
    while (phi >  M_PI)      phi -= 2*M_PI;
    idx_phi = std::round(phi / phi_step);
    idx_dist = std::round(dist_ray / distance_step);
  }

  bool operator==(const Histogram1GramKey &rhs) const {
    return idx_phi == rhs.idx_phi && idx_dist == rhs.idx_dist;
  }

  bool operator!=(const Histogram1GramKey &rhs) const {
    return !(*this == rhs);
  }

  bool operator<(const Histogram1GramKey &rhs) const {
    if (idx_phi != rhs.idx_phi) {
      return idx_phi < rhs.idx_phi;
    }
    return idx_dist < rhs.idx_dist;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & idx_phi;
    ar & idx_dist;
  }
};

struct Histogram2GramKey {
  Histogram1GramKey idx_1;
  int idx_do;

  Histogram1GramKey idx_2;

  Histogram2GramKey() :
    idx_1(), idx_do(0), idx_2() {
  }

  Histogram2GramKey(const ModelObservation &mo1, const ModelObservation &mo2, double phi_step, double distance_step, double max_size) :
    Histogram2GramKey(mo1, mo2.phi, mo2.dist_ray, phi_step, distance_step, max_size) {
  }

  Histogram2GramKey(const ModelObservation &mo1, double phi, double dist_ray, double phi_step, double distance_step, double max_size) :
   idx_1(mo1, phi_step, distance_step), idx_2(phi, dist_ray, phi_step, distance_step) {

    double dist_obs = mo1.dist_obs;
    if (dist_obs > max_size) {
      dist_obs = max_size;
    }

    if (dist_obs < -max_size) {
      dist_obs = -max_size;
    }

    idx_do = std::round(dist_obs / distance_step);
  }

  bool operator<(const Histogram2GramKey &rhs) const {
    if (idx_1 != rhs.idx_1) {
      return idx_1 < rhs.idx_1;
    }

    if (idx_do != rhs.idx_do) {
      return idx_do < rhs.idx_do;
    }

    return idx_2 < rhs.idx_2;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & idx_1;
    ar & idx_do;

    ar & idx_2;
  }
};

class RayModel {
 public:
  RayModel();
  RayModel(double size);
  RayModel(double size, double phi_step, double distance_step);

  // Assume x_hits are in order by angle
  void MarkObservationsWorldFrame(const ObjectState &os, const std::vector<Observation> &x_hits);

  double EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const;

  double SampleRange(const ObjectState &os, double sensor_angle) const;
  double SampleRange(const ObjectState &os, const ModelObservation &mo1, double sensor_angle) const;

  std::vector<Observation> SampleObservations(const ObjectState &os, const std::vector<double> &sensor_angles) const;
  std::vector<Observation> SampleObservations(const ObjectState &os, const std::vector<double> &sensor_angles, std::vector<int> *n_gram) const;

  double GetLikelihood(const ObjectState &os, const Observation &obs) const;
  double GetProbability(const ObjectState &os, const Observation &obs) const;

  double GetSize() const;

  void PrintStats() const;

  void UseNGram(int n_gram);

 private:
  const double kMaxRange_ = 100.0;

  double phi_step_ = 0.1; // ~5 degrees
  double distance_step_ = 0.10; // 10 cm

  double max_size_;
  int phi_dim_;
  int dist_dim_;

  int n_gram_ = 1;

  std::map<Histogram1GramKey, Histogram> histograms_1_gram_;
  std::map<Histogram2GramKey, Histogram> histograms_2_gram_;

  void MarkObservation(const ModelObservation &mo);
  void MarkObservations(const ModelObservation &mo1, const ModelObservation &mo2);

  Histogram* GetHistogram(const ModelObservation &mo);
  Histogram* GetHistogram(double phi, double dist_ray);

  Histogram* GetHistogram(const ModelObservation &mo1, const ModelObservation &mo2);
  Histogram* GetHistogram(const ModelObservation &mo1, double phi, double dist_ray);

  const Histogram* GetHistogram(const ModelObservation &mo) const;
  const Histogram* GetHistogram(double phi, double dist_ray) const;

  const Histogram* GetHistogram(const ModelObservation &mo1, const ModelObservation &mo2) const;
  const Histogram* GetHistogram(const ModelObservation &mo1, double phi, double dist_ray) const;

  void ConvertRay(const ObjectState &os, double sensor_angle, double *phi, double *dist_ray) const;

  bool GetLogLikelihood(const ModelObservation &mo, double *res) const;
  bool GetLogLikelihood(const ModelObservation &mo1, const ModelObservation &mo2, double *res) const;

  bool InRoi(const ModelObservation &mo) const;
  bool IsOccluded(const ModelObservation &mo) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & phi_step_;
    ar & distance_step_;

    ar & max_size_;
    ar & phi_dim_;
    ar & dist_dim_;

    ar & histograms_1_gram_;
    ar & histograms_2_gram_;
  }
};
