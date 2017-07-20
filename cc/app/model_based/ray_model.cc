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

RayModel::RayModel(const std::vector<RayModel*> models, const std::vector<double> probs) {
  // Find size
  double size = 0.0;
  for (const auto &m : models) {
    if (m->GetSize() > size) {
      size = m->GetSize();
    }
  }
  max_size_ = size;

  // Initialize
  phi_dim_ = ceil(2*M_PI / kPhiStep_) + 1;
  dist_dim_ = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i < phi_dim_ * dist_dim_; i++) {
    histograms_.emplace_back(-size, size, kDistanceStep_);
  }

  // Now build up
  for (int idx_phi = 0; idx_phi < phi_dim_; idx_phi++) {
    for (int idx_dist = 0; idx_dist < dist_dim_; idx_dist++) {

      // Compute phi and dist
      double phi = (idx_phi * kPhiStep_) + kPhiStep_/2;
      double dist =  (idx_dist - dist_dim_ / 2) * kDistanceStep_ + kDistanceStep_/2;

      // Get histogram
      int idx = GetHistogramIndex(phi, dist);
      Histogram &h = histograms_[idx];

      // For every value in histogram...
      for (double val = h.GetMin() + h.GetRes() / 2; val < h.GetMax(); val += h.GetRes()) {
        // For every model we're combining
        for (size_t i=0; i<models.size(); i++) {
          const RayModel *m = models[i];

          int idx_m = m->GetHistogramIndex(phi, dist);
          if (idx_m < 0) {
            continue;
          }

          const Histogram &h_m = m->histograms_[idx_m];
          //if (val <= h_m.GetMin() || val >= h_m.GetMax()) {
          //  continue;
          //}

          double weight = probs[i] * h_m.GetLikelihood(val);
          //printf("mark %5.3f with weight %7.5f (prob %5.3f)\n", val, weight, probs[i]);
          h.Mark(val, weight);
        }
      }
    }
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

bool RayModel::GetLogLikelihood(double phi, double dist_ray, double dist_obs, double *res) const {
  int idx = GetHistogramIndex(phi, dist_ray);
  if (idx < 0) {
    return false;
  }

  const Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() == 0) {
    return false;
  }

  if (dist_obs < h.GetMin()) {
    return false;
  }

  double l = h.GetLikelihood(dist_obs);
  if (l < 1e-9) {
    l = 1e-9;
  }

  *res = log(l);
  return true;
}

double RayModel::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits, const RayModel &obs_model) const {
  double l_p_z = 0.0;

  for (const Eigen::Vector2d& x_hit : x_hits) {
    double phi, dist_ray, dist_obs;
    ConvertObservation(x_sensor_object, object_angle, x_hit, &phi, &dist_ray, &dist_obs);

    // Check for projected observations behind us
    double range = x_hit.norm();
    if (dist_obs > range) {
      continue;
    }

    double log_l_obs_obj = 0;
    double log_l_obs = 0;

    if ( GetLogLikelihood(phi, dist_ray, dist_obs, &log_l_obs_obj) && obs_model.GetLogLikelihood(phi, dist_ray, dist_obs, &log_l_obs) ) {
      l_p_z += log_l_obs_obj;
      l_p_z -= log_l_obs;
      //printf("update: %5.3f, %5.3f\n", log_l_obs_obj, log_l_obs);
    }
  }

  return l_p_z;
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
