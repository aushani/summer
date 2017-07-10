#include "ray_model.h"

#include <math.h>

RayModel::RayModel(double size) :
 max_size_(size) {
  angle_dim_ = ceil(2*M_PI / kAngleStep_) + 1;
  dist_dim_ = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i < angle_dim_ * dist_dim_; i++) {
    histograms_.emplace_back(-size, size, kDistanceStep_);
  }
}

void RayModel::MarkObservation(double angle, double dist_line, double dist_obs) {
  int idx = GetHistogramIndex(angle, dist_line);
  if (idx < 0)
    return;

  Histogram &h = histograms_[idx];
  if (h.InRange(dist_obs)) {
    h.Mark(dist_obs);
  }
}

void RayModel::MarkObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double angle, dist_line, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);
  MarkObservation(angle, dist_line, dist_obs);
}

double RayModel::EvaluateObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  bool print = fabs(x_sensor_object(0) - 3) < 1e-3 && fabs(x_sensor_object(1) - 3) < 1e-3;

  double angle, dist_line, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);

  if (print) {
    printf("\n");
    printf("x_sensor_object: %5.3f, %5.3f\n", x_sensor_object(0), x_sensor_object(1));
    printf("x_hit: %5.3f, %5.3f\n", x_hit(0), x_hit(1));
    printf("sensor angle: %5.3f deg\n", angle * 180.0 / M_PI);
    printf("sensor dist line: %5.3f\n", dist_line);
    printf("sensor dist obs: %5.3f\n", dist_obs);
  }

  int idx = GetHistogramIndex(angle, dist_line);
  if (print) {
    printf("histogram index: %d\n", idx);
  }

  if (idx < 0)
    return 0.0;

  Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() < 1)
    return 0.0;

  double cdf = h.GetCumulativeProbability(dist_obs);

  if (print) {
    printf("cdf: %5.3f\n", cdf);
  }

  if (cdf < 0.2 || cdf > 0.8)
    return -1.0;
  return 1.0;
}

void RayModel::ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *angle, double *dist_line, double *dist_obs) {
  // Compute relative angle with object
  double sensor_angle = atan2(x_hit(1), x_hit(0));
  *angle = object_angle - sensor_angle;

  // Compute ray's distance from object center
  double a = x_hit(0)/x_hit.norm();
  double b = x_hit(1)/x_hit.norm();
  *dist_line = a*x_sensor_object(0) + b*x_sensor_object(1);

  // Compute location of ray hit relative to object
  double a_obj = -b;
  double b_obj = a;
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  *dist_obs = a_obj*x_hit(0) + b_obj*x_hit(1) + c_obj;
}

int RayModel::GetHistogramIndex(double angle, double dist_line) {
  while (angle < 0) angle += 2*M_PI;
  while (angle > 2*M_PI) angle -= 2*M_PI;
  int idx_angle = round(angle / kAngleStep_);

  int idx_dist = round(dist_line / kDistanceStep_) + dist_dim_ / 2;
  if (idx_dist < 0 || idx_dist >= dist_dim_)
    return -1;

  return idx_angle * dist_dim_ + idx_dist;
}
