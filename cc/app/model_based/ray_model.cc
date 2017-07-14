#include "ray_model.h"

#include <math.h>

RayModel::RayModel(double size) :
 max_size_(size) {
  phi_dim_ = ceil(2*M_PI / kPhiStep_) + 1;
  dist_dim_ = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i < phi_dim_ * dist_dim_; i++) {
    histograms_.emplace_back(-size, size, kDistanceStep_);
    negative_histograms_.emplace_back(-size, size, kDistanceStep_);
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
  //printf("Mark histogram %d phi %5.3f dist line %5.3f dist obs %5.3f dist_obs\n", idx, phi * 180.0 / M_PI, dist_ray, dist_obs);
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

void RayModel::MarkNegativeObservation(double phi, double dist_ray, double dist_obs) {
  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0)
    return;

  Histogram &h = negative_histograms_[idx];
  h.Mark(dist_obs);
}

void RayModel::MarkNegativeObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double phi, dist_ray, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

  // We are between object and hit
  double range = x_hit.norm();
  if (dist_obs > range)
    return;

  MarkNegativeObservation(phi, dist_ray, dist_obs);
}

double RayModel::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) {
  double s_obj = 0.0;
  double s_noobj = 0.0;

  for (const Eigen::Vector2d& x_hit : x_hits) {
    double phi, dist_ray, dist_obs;
    ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

    double range = x_hit.norm();
    if (dist_obs > range)
      continue;

    int idx = GetHistogramIndex(phi, dist_ray);
    if (idx < 0)
      continue;

    Histogram &h = histograms_[idx];
    Histogram &h_neg = negative_histograms_[idx];
    if (h.GetCountsTotal() < 1 || h_neg.GetCountsTotal() < 1)
      continue;

    double p_obs_obj = h.GetLikelihood(dist_obs);
    double p_obs_noobj = h_neg.GetLikelihood(dist_obs);

    // Occlusion doesn't tell us anything
    // assume p_obs_obj = p_obs_noobj when occluded
    if (dist_obs < h.GetMin()) {
      continue;
    }

    if (p_obs_obj < 1e-9)
      p_obs_obj = 1e-9;
    if (p_obs_noobj < 1e-9)
      p_obs_noobj = 1e-9;

    s_obj += log(p_obs_obj);
    s_noobj += log(p_obs_noobj);
  }

  return s_obj - s_noobj;
}

void RayModel::ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *phi, double *dist_ray, double *dist_obs) {
  // Compute relative angle with object
  double sensor_angle = atan2(x_hit(1), x_hit(0));
  *phi = sensor_angle - object_angle;

  // Compute ray's distance from object center
  double a = x_hit(1)/x_hit.norm();
  double b = -x_hit(0)/x_hit.norm();
  *dist_ray = a*x_sensor_object(0) + b*x_sensor_object(1);

  // Compute location of ray hit relative to object
  double a_obj = -b;
  double b_obj = a;
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  *dist_obs = a_obj*x_hit(0) + b_obj*x_hit(1) + c_obj;
}

double RayModel::GetExpectedRange(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double percentile) {
  //printf("\n");

  // Compute relative angle with object
  double phi = sensor_angle - object_angle;

  // Compute ray's distance from object center
  double a = sin(sensor_angle);
  double b = -cos(sensor_angle);
  double dist_ray = a*x_sensor_object(0) + b*x_sensor_object(1);
  //printf("Angle: %5.3f, dist line: %5.3f\n", angle*180.0/M_PI, dist_ray);
  //printf("Line is: %5.3f x + %5.3f y + 0 = 0\n", a, b);

  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0) {
    //printf("phi %5.3f dist line %5.3f out of range\n", phi*180.0/M_PI, dist_ray);
    return 100.0;
  }

  Histogram &h = histograms_[idx];
  //printf("Histogram %d count: %d\n", idx, h.GetCountsTotal());
  if (h.GetCountsTotal() == 0) {
    //printf("phi %5.3f dist line %5.3f has no measurements\n", phi*180.0/M_PI, dist_ray);
    return 0.0;
  }

  double dist_obs = h.GetPercentile(percentile);
  //printf("%5.3f%% dist obs: %5.3f\n", percentile*100.0f, dist_obs);
  if (dist_obs >= h.GetMax()) {
    //printf("phi %5.3f dist line %5.3f at the end\n", phi*180.0/M_PI, dist_ray);
    return 100.0;
  }

  double a_obj = -b;
  double b_obj = a;
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  double dist_sensor = a_obj*0 * b_obj*0 + c_obj;

  double range = dist_obs - dist_sensor;
  if (range < 0) {
    //printf("phi %5.3f dist line %5.3f too close\n", phi*180.0/M_PI, dist_ray);
    range = 0;
  }

  //printf("Dist sensor: %5.3f\n", dist_sensor);
  //printf("Range: %5.3f\n", range);

  return range;
}

double RayModel::GetProbability(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double phi, dist_ray, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

  printf("dist obs: %5.3f\n", dist_obs);

  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0) {
    return 0.0;
  }

  Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() < 1) {
    return 0.0;
  }

  return h.GetProbability(dist_obs);
}

int RayModel::GetHistogramIndex(double phi, double dist_ray) {
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
