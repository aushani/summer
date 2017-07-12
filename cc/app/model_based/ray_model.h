#pragma once

#include <vector>
#include <stdio.h>
#include <cstddef>

#include <Eigen/Core>

#include "histogram.h"

class RayModel {
 public:
  RayModel(double size);

  void MarkObservation(double angle, double dist_line, double dist_obs);
  void MarkObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

  void MarkNegativeObservation(double angle, double dist_line, double dist_obs);
  void MarkNegativeObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

  double EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits);

  double GetExpectedRange(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double percentile);
  double GetProbability(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

 private:
  const double kAngleStep_ = 0.1;
  const double kDistanceStep_ = 0.1;

  double max_size_;
  int angle_dim_;
  int dist_dim_;

  std::vector<Histogram> histograms_;
  std::vector<Histogram> negative_histograms_;

  int GetHistogramIndex(double angle, double dist_line);
  void ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *angle, double *dist_line, double *dist_obs);
};
