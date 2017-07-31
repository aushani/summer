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

class EmptyModel {
 public:
  EmptyModel();

  void MarkObservationWorldFrame(const ObjectState &os, const Observation &x_hit);

  double EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const;

  double GetLikelihood(const ObjectState &os, const Observation &x_hit) const;

  void PrintStats() const;

 private:
  double kDistanceStep_ = 0.15; // 15 cm

  double max_size_ = 5.0;

  std::vector<Histogram> histograms_;

  void MarkObservation(double dist_ray, double dist_obs);

  int GetHistogramIndex(double dist_ray) const;
  void ConvertObservation(const ObjectState &os, const Observation &x_hit, double *dist_ray, double *dist_obs) const;

  bool GetLogLikelihood(double dist_ray, double dist_obs, double *res) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & kDistanceStep_;
    ar & max_size_;

    ar & histograms_;
  }
};
