#pragma once

#include <vector>
#include <stdio.h>
#include <cstddef>

#include <Eigen/Core>

#include "histogram.h"

class RayModel {
 public:
  RayModel(double size);

  void MarkObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

  double EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const;

  double GetExpectedRange(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double percentile) const;

  double GetSize() const;

 private:
  double kPhiStep_ = 0.05;
  double kDistanceStep_ = 0.3;
  double kMaxRange_ = 100.0;

  double max_size_;
  int phi_dim_;
  int dist_dim_;

  std::vector<Histogram> histograms_;

  void MarkObservation(double phi, double dist_ray, double dist_obs);

  int GetHistogramIndex(double phi, double dist_ray) const;
  void ConvertRay(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double *phi, double *dist_ray) const;
  void ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *phi, double *dist_ray, double *dist_obs) const;

  bool GetLogLikelihood(double phi, double dist_ray, double dist_obs, double *res) const;
};
