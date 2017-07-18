#include "ray_model.h"

#include <math.h>

RayModel::RayModel(double size) :
 max_size_(size) {
  phi_dim_ = ceil(2*M_PI / kPhiStep_) + 1;
  dist_dim_ = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i < phi_dim_ * dist_dim_; i++) {
    histograms_.emplace_back(-size, size, kDistanceStep_);
  }
}

double RayModel::GetSize() const {
  return max_size_;
}

void RayModel::MarkObservation(double phi, double dist_ray, double dist_obs) {
  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0)
    return;

  Histogram &h = histograms_[idx];
  h.Mark(dist_obs);
}

void RayModel::MarkObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double phi, dist_ray, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

  // We are between object and hit
  double range = x_hit.norm();
  if (dist_obs > range)
    return;

  MarkObservation(phi, dist_ray, dist_obs);
}

double RayModel::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const {
  double s_obj = 0.0;

  for (const Eigen::Vector2d& x_hit : x_hits) {
    double phi, dist_ray, dist_obs;
    ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

    // Check for projected observations behind us
    double range = x_hit.norm();
    if (dist_obs > range) {
      continue;
    }

    int idx = GetHistogramIndex(phi, dist_ray);
    if (idx < 0) {
      continue;
    }

    const Histogram &h = histograms_[idx];
    if (h.GetCountsTotal() < 1) {
      continue;
    }

    double p_obs_obj = h.GetLikelihood(dist_obs);

    // Occlusion doesn't tell us anything
    // assume p_obs_obj = p_obs_noobj when occluded
    if (dist_obs < h.GetMin()) {
      continue;
    }

    if (p_obs_obj < 1e-9)
      p_obs_obj = 1e-9;

    s_obj += log(p_obs_obj);
  }

  return s_obj;
}

void RayModel::ConvertRay(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double *phi, double *dist_ray) const {
  // Compute relative angle with object
  *phi = sensor_angle - object_angle;

  // Compute ray's distance from object center
  double a = sin(sensor_angle);
  double b = -cos(sensor_angle);
  *dist_ray = a*x_sensor_object(0) + b*x_sensor_object(1);
}

void RayModel::ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *phi, double *dist_ray, double *dist_obs) const {
  // Compute relative angle with object
  double sensor_angle = atan2(x_hit(1), x_hit(0));
  ConvertRay(x_sensor_object, object_angle, sensor_angle, phi, dist_ray);

  // Compute location of ray hit relative to object
  double a_obj = cos(sensor_angle);
  double b_obj = sin(sensor_angle);
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  *dist_obs = a_obj*x_hit(0) + b_obj*x_hit(1) + c_obj;
}

double RayModel::GetExpectedRange(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double percentile) const {
  // Compute relative angle with object
  double phi=0.0, dist_ray=0.0;
  ConvertRay(x_sensor_object, object_angle, sensor_angle, &phi, &dist_ray);

  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0) {
    return kMaxRange_;
  }

  const Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() == 0) {
    return kMaxRange_;
  }

  double dist_obs = h.GetPercentile(percentile);

  double a_obj = cos(sensor_angle);
  double b_obj = sin(sensor_angle);
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  double dist_sensor = c_obj;

  double range = dist_obs - dist_sensor;
  if (range < 0) {
    range = 0;
  }

  return range;
}

int RayModel::GetHistogramIndex(double phi, double dist_ray) const {
  while (phi < 0) phi += 2*M_PI;
  while (phi > 2*M_PI) phi -= 2*M_PI;
  int idx_phi = round(phi / kPhiStep_);

  int idx_dist = round(dist_ray / kDistanceStep_) + dist_dim_ / 2;
  if (idx_dist < 0 || idx_dist >= dist_dim_) {
    //printf("dist %5.3f out of range (dim: %d)\n", dist_ray, dist_dim_);
    return -1;
  }

  return idx_phi * dist_dim_ + idx_dist;
}
