#include "empty_model.h"

EmptyModel::EmptyModel() {
  int n_hists = 2*ceil(max_size_ / kDistanceStep_) + 1;

  for (int i = 0; i<n_hists; i++) {
    histograms_.emplace_back(-max_size_, max_size_, kDistanceStep_);
  }
}

void EmptyModel::MarkObservation(double dist_ray, double dist_obs) {
  int idx = GetHistogramIndex(dist_ray);
  if (idx < 0)
    return;

  histograms_[idx].Mark(dist_obs);
}

void EmptyModel::MarkObservationWorldFrame(const ObjectState &os, const Observation &x_hit) {
  double dist_ray, dist_obs;
  ConvertObservation(os, x_hit, &dist_ray, &dist_obs);

  // We are between object and hit
  double range = x_hit.GetRange();
  if (dist_obs > range) {
    return;
  }

  MarkObservation(dist_ray, dist_obs);
}

bool EmptyModel::GetLogLikelihood(double dist_ray, double dist_obs, double *res) const {
  int idx = GetHistogramIndex(dist_ray);
  if (idx < 0) {
    return false;
  }

  const Histogram &h = histograms_[idx];
  if (h.GetCountsTotal() < 10) {
    printf("Count = %5.3f, histogram with dist_ray = %5.3f!!!\n", h.GetCountsTotal(), dist_ray);
    return false;
  }

  // Occluded
  if (dist_obs < h.GetMin()) {
    return false;
  }

  double l = h.GetLikelihood(dist_obs);
  if (l < 1e-99) {
    l = 1e-99;
  }

  *res = log(l);
  return true;
}

double EmptyModel::EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const {
  double l_p_z = 0.0;

  const double angle_to_object = os.GetBearing();

  double max_dtheta = os.GetMaxDtheta();

  for (const auto &x_hit : x_hits) {
    if (max_dtheta < 2*M_PI) {
      float angle = x_hit.GetTheta();

      double dtheta = angle - angle_to_object;
      while (dtheta < -M_PI) dtheta += 2*M_PI;
      while (dtheta > M_PI) dtheta -= 2*M_PI;

      if (std::abs(dtheta) > max_dtheta) {
        //// Verify
        //double phi, dist_ray, dist_obs;
        //ConvertObservation(os, x_hit, &phi, &dist_ray, &dist_obs);

        //// Check for projected observations behind us
        //double range = x_hit.GetRange();
        //if (dist_obs > range) {
        //  continue;
        //}

        //double log_l_obs_obj = 0;
        //if ( GetLogLikelihood(phi, dist_ray, dist_obs, &log_l_obs_obj) ) {
        //  printf("ERROR ERROR ERROR, %5.3f dtheta vs %5.3f max dtheta\n",
        //      dtheta, max_dtheta);
        //}
        continue;
      }
    }

    double dist_ray, dist_obs;
    ConvertObservation(os, x_hit, &dist_ray, &dist_obs);

    // Check for projected observations behind us
    double range = x_hit.GetRange();
    if (dist_obs > range) {
      continue;
    }

    double log_l_obs_obj = 0;

    if ( GetLogLikelihood(dist_ray, dist_obs, &log_l_obs_obj) ) {
      l_p_z += log_l_obs_obj;
      //printf("update: %5.3f, %5.3f\n", log_l_obs_obj, log_l_obs);
    }
  }

  return l_p_z;
}

void EmptyModel::ConvertObservation(const ObjectState &os, const Observation &x_hit, double *dist_ray, double *dist_obs) const {
  // Compute ray's distance from object center
  *dist_ray = x_hit.GetSinTheta()*os.GetPos()(0) - x_hit.GetCosTheta()*os.GetPos()(1);

  // Compute location of ray hit relative to object
  *dist_obs = x_hit.GetRange() - x_hit.GetCosTheta()*os.GetPos()(0) - x_hit.GetSinTheta()*os.GetPos()(1);
}

int EmptyModel::GetHistogramIndex(double dist_ray) const {
  int idx_dist = round(dist_ray / kDistanceStep_) + histograms_.size() / 2;
  if (idx_dist < 0 || idx_dist >= histograms_.size()) {
    //printf("dist %5.3f out of range (dim: %d)\n", dist_ray, dist_dim_);
    return -1;
  }

  return idx_dist;
}

void EmptyModel::PrintStats() const {
  double min_l = 100000.0;
  double max_l = -100000.0;
  double min_count = 9999999;
  double max_count = 0;
  for (const auto& h : histograms_) {
    for (double x = h.GetMin(); x <= h.GetMax(); x+=h.GetRes()) {
      double l = h.GetLikelihood(x);

      if (l == 0)
        continue;

      if (l < min_l) min_l = l;
      if (l > max_l) max_l = l;

      if (h.GetCountsTotal() > max_count) max_count = h.GetCountsTotal();
      if (h.GetCountsTotal() < min_count) min_count = h.GetCountsTotal();
    }
  }

  printf("\tLikelihood ranges from %f to %f\n", min_l, max_l);
  printf("\tCount ranges from %f to %f\n", min_count, max_count);
}

double EmptyModel::GetLikelihood(const ObjectState &os, const Observation &x_hit) const {
  double dist_ray, dist_obs;
  ConvertObservation(os, x_hit, &dist_ray, &dist_obs);

  double range = x_hit.GetRange();
  if (dist_obs > range) {
    return -1.0;
  }

  int idx = GetHistogramIndex(dist_ray);
  if (idx < 0) {
    return -1.0;
  }

  return histograms_[idx].GetLikelihood(dist_obs);
}
