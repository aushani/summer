#pragma once

#include <vector>
#include <stdio.h>
#include <cstddef>

#include <Eigen/Core>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "histogram.h"
#include "observation.h"
#include "object_state.h"

class RayModel {
 public:
  RayModel();
  RayModel(double size);

  void MarkObservationWorldFrame(const ObjectState &os, const Observation &x_hit);

  double EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const;

  double SampleRange(const ObjectState &os, double sensor_angle) const;
  double GetExpectedRange(const ObjectState &os, double sensor_angle, double percentile) const;
  double GetLikelihood(const ObjectState &os, const Observation &x_hit) const;

  double GetSize() const;

  void PrintStats() const;

 private:
  double kPhiStep_ = 0.1; // ~0.5 degrees
  double kDistanceStep_ = 0.15; // 15 cm
  double kMaxRange_ = 100.0;

  double max_size_;
  int phi_dim_;
  int dist_dim_;

  std::vector<Histogram> histograms_;

  void MarkObservation(double phi, double dist_ray, double dist_obs);

  int GetHistogramIndex(double phi, double dist_ray) const;
  void ConvertRay(const ObjectState &os, double sensor_angle, double *phi, double *dist_ray) const;
  void ConvertObservation(const ObjectState &os, const Observation &x_hit, double *phi, double *dist_ray, double *dist_obs) const;

  bool GetLogLikelihood(double phi, double dist_ray, double dist_obs, double *res) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & kPhiStep_;
    ar & kDistanceStep_;
    ar & kMaxRange_;

    ar & max_size_;
    ar & phi_dim_;
    ar & dist_dim_;

    ar & histograms_;
  }
};
