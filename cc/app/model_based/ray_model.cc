#include "ray_model.h"

#include <math.h>

RayModel::RayModel(double size) :
 max_size_(size) {
  angle_dim_ = ceil(2*M_PI / kAngleStep_) + 1;
  dist_dim_ = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i < angle_dim_ * dist_dim_; i++) {
    histograms_.emplace_back(-size, size, kDistanceStep_);
    negative_histograms_.emplace_back(-size, size, kDistanceStep_);
  }
}

void RayModel::MarkObservation(double angle, double dist_line, double dist_obs) {
  int idx = GetHistogramIndex(angle, dist_line);
  if (idx < 0)
    return;

  Histogram &h = histograms_[idx];
  h.Mark(dist_obs);
  //printf("Mark histogram %d angle %5.3f dist line %5.3f dist obs %5.3f dist_obs\n", idx, angle * 180.0 / M_PI, dist_line, dist_obs);
}

void RayModel::MarkObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double angle, dist_line, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);

  // We are between object and hit
  double range = x_hit.norm();
  if (dist_obs > range)
    return;

  MarkObservation(angle, dist_line, dist_obs);
}

void RayModel::MarkNegativeObservation(double angle, double dist_line, double dist_obs) {
  int idx = GetHistogramIndex(angle, dist_line);
  if (idx < 0)
    return;

  Histogram &h = negative_histograms_[idx];
  h.Mark(dist_obs);
}

void RayModel::MarkNegativeObservationWorldFrame(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double angle, dist_line, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);

  // We are between object and hit
  double range = x_hit.norm();
  if (dist_obs > range)
    return;

  MarkNegativeObservation(angle, dist_line, dist_obs);
}

double RayModel::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) {
  double sum_log_p_obj = 0.0;
  double sum_log_p_noobj = 0.0;

  double p_obj = 0.5;
  double p_noobj = 1 - p_obj;

  sum_log_p_obj += log(p_obj);
  sum_log_p_noobj += log(p_noobj);

  for (const Eigen::Vector2d& x_hit : x_hits) {
    double angle, dist_line, dist_obs;
    ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);

    double range = x_hit.norm();
    if (dist_obs > range)
      continue;

    int idx = GetHistogramIndex(angle, dist_line);
    if (idx < 0)
      continue;

    Histogram &h = histograms_[idx];
    Histogram &h_neg = negative_histograms_[idx];
    if (h.GetCountsTotal() < 1 || h_neg.GetCountsTotal() < 1)
      continue;

    double p_ray_obj = h.GetLikelihood(dist_obs);
    double p_ray_noobj = h_neg.GetLikelihood(dist_obs);

    // Occlusion doesn't tell us anything
    // assume p_ray_obj = p_ray_noobj when occluded
    if (dist_obs < h.GetMin()) {
      //printf("\n");
      //printf("Angle %5.3f Dist Line %5.3f\n", angle * 180.0/M_PI, dist_line);
      //printf("Occlusion at dist obs %5.3f: %5.3f vs %5.3f\n", dist_obs, p_ray_obj, p_ray_noobj);
      continue;
    }

    if (p_ray_obj < 1e-9)
      p_ray_obj = 1e-9;
    if (p_ray_noobj < 1e-9)
      p_ray_noobj = 1e-9;

    //printf("dist_obs: %5.3f, %f vs %f\n", dist_obs, p_ray_obj, p_ray_noobj);
    //printf("median: %5.3f, %5.3f\n", h.GetMedian(), h_neg.GetMedian());
    //printf("counts: %d, %d\n", h.GetCountsTotal(), h_neg.GetCountsTotal());

    sum_log_p_obj += log(p_ray_obj);
    sum_log_p_noobj += log(p_ray_noobj);
  }

  //printf("%f vs %f\n", sum_log_p_obj, sum_log_p_noobj);

  return sum_log_p_obj - sum_log_p_noobj;

  // For numerics
  //double avg_log = (sum_log_p_obj + sum_log_p_noobj) / 2;
  //sum_log_p_obj -= avg_log;
  //sum_log_p_noobj -= avg_log;

  //double p_obj_ray = p_ray_obj;
  //double p_noobj_ray = p_ray_noobj;

  //return log(p_obj_ray / p_noobj_ray);
}

void RayModel::ConvertObservation(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit, double *angle, double *dist_line, double *dist_obs) {
  // Compute relative angle with object
  double sensor_angle = atan2(x_hit(1), x_hit(0));
  *angle = sensor_angle - object_angle;

  // Compute ray's distance from object center
  double a = x_hit(1)/x_hit.norm();
  double b = -x_hit(0)/x_hit.norm();
  *dist_line = a*x_sensor_object(0) + b*x_sensor_object(1);

  // Compute location of ray hit relative to object
  double a_obj = -b;
  double b_obj = a;
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  *dist_obs = a_obj*x_hit(0) + b_obj*x_hit(1) + c_obj;
}

double RayModel::GetExpectedRange(const Eigen::Vector2d &x_sensor_object, double object_angle, double sensor_angle, double percentile) {
  //printf("\n");

  // Compute relative angle with object
  double angle = sensor_angle - object_angle;

  // Compute ray's distance from object center
  double a = sin(sensor_angle);
  double b = -cos(sensor_angle);
  double dist_line = a*x_sensor_object(0) + b*x_sensor_object(1);
  //printf("Angle: %5.3f, dist line: %5.3f\n", angle*180.0/M_PI, dist_line);
  //printf("Line is: %5.3f x + %5.3f y + 0 = 0\n", a, b);

  int idx = GetHistogramIndex(angle, dist_line);
  if (idx < 0) {
    //printf("Angle %5.3f dist line %5.3f out of range\n", angle*180.0/M_PI, dist_line);
    return 100.0;
  }

  Histogram &h = histograms_[idx];
  //printf("Histogram %d count: %d\n", idx, h.GetCountsTotal());
  if (h.GetCountsTotal() == 0) {
    //printf("Angle %5.3f dist line %5.3f has no measurements\n", angle*180.0/M_PI, dist_line);
    return 0.0;
  }

  double dist_obs = h.GetPercentile(percentile);
  //printf("%5.3f%% dist obs: %5.3f\n", percentile*100.0f, dist_obs);
  if (dist_obs >= h.GetMax()) {
    //printf("Angle %5.3f dist line %5.3f at the end\n", angle*180.0/M_PI, dist_line);
    return 100.0;
  }

  double a_obj = -b;
  double b_obj = a;
  double c_obj = -(a_obj * x_sensor_object(0) + b_obj * x_sensor_object(1));
  double dist_sensor = a_obj*0 * b_obj*0 + c_obj;

  double range = dist_obs - dist_sensor;
  if (range < 0) {
    //printf("Angle %5.3f dist line %5.3f too close\n", angle*180.0/M_PI, dist_line);
    range = 0;
  }

  //printf("Dist sensor: %5.3f\n", dist_sensor);
  //printf("Range: %5.3f\n", range);

  return range;
}

double RayModel::GetProbability(const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  double angle, dist_line, dist_obs;
  ConvertObservation(x_sensor_object, object_angle, x_hit, &angle, &dist_line, &dist_obs);

  printf("dist obs: %5.3f\n", dist_obs);

  int idx = GetHistogramIndex(angle, dist_line);
  if (idx < 0) {
    return 0.0;
  }

  Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() < 1) {
    return 0.0;
  }

  return h.GetProbability(dist_obs);
}

int RayModel::GetHistogramIndex(double angle, double dist_line) {
  while (angle < 0) angle += 2*M_PI;
  while (angle > 2*M_PI) angle -= 2*M_PI;
  int idx_angle = round(angle / kAngleStep_);

  int idx_dist = round(dist_line / kDistanceStep_) + dist_dim_ / 2;
  if (idx_dist < 0 || idx_dist >= dist_dim_) {
    //printf("dist %5.3f out of range (dim: %d)\n", dist_line, dist_dim_);
    return -1;
  }

  return idx_angle * dist_dim_ + idx_dist;
}
