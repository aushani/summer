#include "ray_model.h"

#include <math.h>

RayModel::RayModel() :
 RayModel(0.0) {
}

RayModel::RayModel(double size) :
 max_size_(size) {
}

RayModel::RayModel(double size, double phi_step, double distance_step) :
 phi_step_(phi_step), distance_step_(distance_step), max_size_(size) {
}

double RayModel::GetSize() const {
  return max_size_;
}

void RayModel::MarkObservation(const ModelObservation &mo) {
  Histogram *h = GetHistogram(mo);
  if (h == nullptr) {
    return;
  }

  h->Mark(mo.dist_obs);
}

void RayModel::MarkObservations(const ModelObservation &mo1, const ModelObservation &mo2) {
  Histogram *h = GetHistogram(mo1, mo2);
  if (h == nullptr) {
    return;
  }

  h->Mark(mo2.dist_obs);
}

bool RayModel::InRoi(const ModelObservation &mo) const {
  // Check if ray comes closes enough to object
  if (std::abs(mo.dist_ray) > max_size_) {
    return false;
  }

  // Check for projected observations behind us
  if (!mo.in_front) {
    return false;
  }

  return true;
}

bool RayModel::IsOccluded(const ModelObservation &mo) const {
  // Check for occlusion
  if (mo.dist_obs < -max_size_) {
    return true;
  }

  return false;
}

void RayModel::MarkObservationsWorldFrame(const ObjectState &os, const std::vector<Observation> &x_hits) {
  // Convert to model observations
  std::vector<ModelObservation> obs;
  for (const Observation &x_hit : x_hits) {
    obs.emplace_back(os, x_hit);
  }

  // Do 1 grams
  for (const ModelObservation &mo : obs) {
    if (InRoi(mo)) {
      MarkObservation(mo);
    }
  }

  // Do 2 grams
  for (size_t i = 0; i < obs.size(); i++) {
    const ModelObservation &mo1 = obs[i==0 ? (obs.size()-1):(i-1)];
    const ModelObservation &mo2 = obs[i];

    if (InRoi(mo1) && InRoi(mo2)) {
      // TODO Need to do two-sided?
      MarkObservations(mo1, mo2);
      MarkObservations(mo2, mo1);
    }
  }
}

bool RayModel::GetLogLikelihood(const ModelObservation &mo, double *res) const {
  const Histogram *h = GetHistogram(mo);
  if (h == nullptr) {
    return false;
  }

  if (h->GetCountsTotal() < 10) {
    printf("Count = %5.3f, histogram with phi = %5.3f, dist_ray = %5.3f!!!\n", h->GetCountsTotal(), mo.phi, mo.dist_ray);
    return false;
  }

  double l = h->GetLikelihood(mo.dist_obs);
  //double l = h->GetProbability(mo.dist_obs);
  if (l < 1e-99) {
    l = 1e-99;
  }

  *res = log(l);
  return true;
}

bool RayModel::GetLogLikelihood(const ModelObservation &mo1, const ModelObservation &mo2, double *res) const {
  double l_min = 1e-99;

  const Histogram *h = GetHistogram(mo1, mo2);
  if (h == nullptr) {
    *res = log(l_min);
    return false;
  }

  if (h->GetCountsTotal() < 10) {
    //printf("Count = %5.3f, histogram with phi_1 = %5.3f, dist_ray_1 = %5.3f, dist_obs_1 = %5.3f, phi_2 = %5.3f, dist_ray_2 = %5.3f,!!!\n",
    //    h->GetCountsTotal(), mo1.phi, mo1.dist_ray, mo1.dist_obs, mo2.phi, mo2.dist_ray);
    *res = log(l_min);
    return false;
  }

  double l = h->GetLikelihood(mo2.dist_obs);
  //double l = h->GetProbability(mo2.dist_obs);
  if (l < l_min) {
    *res = log(l_min);
    return false;
  }

  *res = log(l);
  return true;
}

double RayModel::GetLikelihood(const ObjectState &os, const Observation &obs) const {
  ModelObservation mo(os, obs);

  if (!InRoi(mo)) {
    return -1.0;
  }

  const Histogram *h = GetHistogram(mo);
  if (h == nullptr) {
    return -1.0;
  }

  return h->GetLikelihood(mo.dist_obs);
}

double RayModel::GetProbability(const ObjectState &os, const Observation &obs) const {
  ModelObservation mo(os, obs);

  if (!InRoi(mo)) {
    return -1.0;
  }

  const Histogram *h = GetHistogram(mo);
  if (h == nullptr) {
    return -1.0;
  }

  return h->GetProbability(mo.dist_obs);
}

double RayModel::EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const {
  // Convert to model observations
  std::vector<ModelObservation> obs;
  for (const Observation &x_hit : x_hits) {
    obs.emplace_back(os, x_hit);
  }

  // Find where to start
  // TODO this is hacky
  int offset = 0;
  if (InRoi(obs[0])) {
    offset = obs.size()/2;
  }

  double l_p_z = 0.0;

  for (size_t i = 0; i < obs.size(); i++) {
    const ModelObservation &mo = obs[(i+offset)%obs.size()];

    if (!InRoi(mo)) {
      continue;
    }

    if (IsOccluded(mo)) {
      continue;
    }

    double log_l_obs_obj = 0;
    bool valid = false;

    if (i==0) {
      valid = GetLogLikelihood(mo, &log_l_obs_obj);
    } else {
      const ModelObservation &mo2 = mo;
      const ModelObservation &mo1 = obs[(i-1+offset)%obs.size()];

      if (n_gram_ == 2 && InRoi(mo1) && !IsOccluded(mo1)) {
        valid = GetLogLikelihood(mo1, mo2, &log_l_obs_obj);
      } else {
        valid = GetLogLikelihood(mo, &log_l_obs_obj);
      }
    }

    if (valid) {
      l_p_z += log_l_obs_obj;
    }
  }

  return l_p_z;
}

void RayModel::ConvertRay(const ObjectState &os, double sensor_angle, double *phi, double *dist_ray) const {
  // Compute relative angle with object
  *phi = sensor_angle - os.GetTheta();

  // Compute ray's distance from object center
  double a = sin(sensor_angle);
  double b = -cos(sensor_angle);
  *dist_ray = a*os.GetPos()(0) + b*os.GetPos()(1);
}

double RayModel::SampleRange(const ObjectState &os, double sensor_angle) const {
  double phi, dist_ray;
  ConvertRay(os, sensor_angle, &phi, &dist_ray);

  const Histogram *h = GetHistogram(phi, dist_ray);
  if (h == nullptr) {
    return -1.0;
  }

  if (h->GetCountsTotal() == 0) {
    return -1.0;
  }

  double dist_obs = h->Sample();

  double a_obj = cos(sensor_angle);
  double b_obj = sin(sensor_angle);
  double c_obj = -(a_obj * os.GetPos()(0) + b_obj * os.GetPos()(1));
  double dist_sensor = c_obj;

  double range = dist_obs - dist_sensor;
  if (range < 0) {
    return -1.0;
  }

  return range;
}

std::vector<Observation> RayModel::SampleObservations(const ObjectState &os, const std::vector<double> &sensor_angles) const {
  std::vector<Observation> res;

  for (double sensor_angle : sensor_angles) {
    double range = SampleRange(os, sensor_angle);
    if (range > 0) {
      res.emplace_back(range, sensor_angle);
    }
  }

  return res;
}

double RayModel::SampleRange(const ObjectState &os, const ModelObservation &mo1, double sensor_angle) const {
  double phi, dist_ray;
  ConvertRay(os, sensor_angle, &phi, &dist_ray);

  const Histogram *h = GetHistogram(mo1, phi, dist_ray);
  if (h == nullptr) {
    return -1.0;
  }

  if (h->GetCountsTotal() == 0) {
    return -1.0;
  }

  double dist_obs = h->Sample();

  double a_obj = cos(sensor_angle);
  double b_obj = sin(sensor_angle);
  double c_obj = -(a_obj * os.GetPos()(0) + b_obj * os.GetPos()(1));
  double dist_sensor = c_obj;

  double range = dist_obs - dist_sensor;
  if (range < 0) {
    return -1.0;
  }

  return range;
}

std::vector<Observation> RayModel::SampleObservations(const ObjectState &os, const std::vector<double> &sensor_angles, std::vector<int> *n_gram) const {
  std::vector<Observation> res;

  for (size_t i=0; i < sensor_angles.size(); i++) {
    double sensor_angle = sensor_angles[i];
    double range = -1.0;
    int n = 0;

    double phi, dist_ray;
    ConvertRay(os, sensor_angle, &phi, &dist_ray);

    if (!res.empty()) {
      ModelObservation mo1(os, res.back());
      range = SampleRange(os, mo1, sensor_angle);
      n = 2;
    }

    if (range < 0) {
      range = SampleRange(os, sensor_angle);
      n = 1;
    }

    if (range > 0) {
      res.emplace_back(range, sensor_angle);
      n_gram->push_back(n);
    }
  }

  return res;
}

Histogram* RayModel::GetHistogram(const ModelObservation &mo) {
  return GetHistogram(mo.phi, mo.dist_ray);
}

Histogram* RayModel::GetHistogram(double phi, double dist_ray) {
  if (std::abs(dist_ray) > max_size_) {
    return nullptr;
  }

  Histogram1GramKey key(phi, dist_ray, phi_step_, distance_step_);

  // insert if not there
  if (histograms_1_gram_.count(key) == 0) {
    Histogram hist(-max_size_, max_size_, distance_step_);
    histograms_1_gram_.insert( std::pair<Histogram1GramKey, Histogram>(key, hist) );
  }

  auto it = histograms_1_gram_.find(key);
  return &it->second;
}

Histogram* RayModel::GetHistogram(const ModelObservation &mo1, const ModelObservation &mo2) {
  return GetHistogram(mo1, mo2.phi, mo2.dist_ray);
}

Histogram* RayModel::GetHistogram(const ModelObservation &mo1, double phi, double dist_ray) {
  if (std::abs(dist_ray) > max_size_) {
    return nullptr;
  }

  if (std::abs(mo1.dist_ray) > max_size_) {
    return nullptr;
  }

  //if (std::abs(mo1.dist_obs) > max_size_) {
  //  return nullptr;
  //}

  Histogram2GramKey key(mo1, phi, dist_ray, phi_step_, distance_step_, max_size_);

  // insert if not there
  if (histograms_2_gram_.count(key) == 0) {
    Histogram hist(-max_size_, max_size_, distance_step_);
    histograms_2_gram_.insert( std::pair<Histogram2GramKey, Histogram>(key, hist) );
  }

  auto it = histograms_2_gram_.find(key);
  return &it->second;
}

const Histogram* RayModel::GetHistogram(const ModelObservation &mo) const {
  return GetHistogram(mo.phi, mo.dist_ray);
}

const Histogram* RayModel::GetHistogram(double phi, double dist_ray) const {
  if (std::abs(dist_ray) > max_size_) {
    return nullptr;
  }

  Histogram1GramKey key(phi, dist_ray, phi_step_, distance_step_);

  // check if not there
  if (histograms_1_gram_.count(key) == 0) {
    return nullptr;
  }

  auto it = histograms_1_gram_.find(key);
  return &it->second;
}

const Histogram* RayModel::GetHistogram(const ModelObservation &mo1, const ModelObservation &mo2) const {
  return GetHistogram(mo1, mo2.phi, mo2.dist_ray);
}

const Histogram* RayModel::GetHistogram(const ModelObservation &mo1, double phi, double dist_ray) const {
  if (std::abs(dist_ray) > max_size_) {
    return nullptr;
  }

  if (std::abs(mo1.dist_ray) > max_size_) {
    return nullptr;
  }

  //if (std::abs(mo1.dist_obs) > max_size_) {
  //  return nullptr;
  //}

  Histogram2GramKey key(mo1, phi, dist_ray, phi_step_, distance_step_, max_size_);

  // check if not there
  if (histograms_2_gram_.count(key) == 0) {
    return nullptr;
  }

  auto it = histograms_2_gram_.find(key);
  return &it->second;
}

void RayModel::PrintStats() const {
  printf("Print stats TODO\n");
  printf("Have %ld 1-gram histograms, %ld 2-gram histograms\n", histograms_1_gram_.size(), histograms_2_gram_.size());
}

void RayModel::UseNGram(int n_gram) {
  n_gram_ = n_gram;
}
