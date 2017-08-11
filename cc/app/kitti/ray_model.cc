#include "app/kitti/ray_model.h"

#include <tuple>

#include <boost/assert.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "library/timer/timer.h"
#include "library/util/angle.h"

namespace ut = library::util;

namespace app {
namespace kitti {

RayModel::RayModel() {

}

bool RayModel::InRoi(const ModelObservation &mo) const {
  // Check if ray comes closes enough to object
  if (std::abs(mo.dist_ray) > max_size_xy_ || std::abs(mo.dist_z) > max_size_z_) {
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
  if (mo.dist_obs < -max_size_xy_) {
    return true;
  }

  return false;
}

void RayModel::MarkObservation(const ModelObservation &mo) {
  library::histogram::Histogram *h = GetHistogram(mo);
  if (h == nullptr) {
    return;
  }

  h->Mark(mo.dist_obs);

  Histogram1GramKey key(mo, kDistRes, kAngleRes, max_size_xy_, max_size_z_);
}

void RayModel::MarkObservations(const ObjectState &os, const std::vector<ModelObservation> &obs) {
  // Do 1 grams
  for (const ModelObservation &mo : obs) {
    if (InRoi(mo)) {
      MarkObservation(mo);
    }
  }
}

library::histogram::Histogram* RayModel::GetHistogram(const ModelObservation &mo) {
  Histogram1GramKey key(mo, kDistRes, kAngleRes, max_size_xy_, max_size_z_);

  if (!key.InRange()) {
    return nullptr;
  }

  // insert if not there
  if (histograms_.count(key) == 0) {
    library::histogram::Histogram hist(-max_size_xy_, max_size_xy_, kDistRes);
    histograms_.insert( std::pair<Histogram1GramKey, library::histogram::Histogram>(key, hist) );
  }

  auto it = histograms_.find(key);
  return &it->second;
}

const library::histogram::Histogram* RayModel::GetHistogram(const ModelObservation &mo) const {
  Histogram1GramKey key(mo, kDistRes, kAngleRes, max_size_xy_, max_size_z_);

  if (!key.InRange() || histograms_.count(key) == 0) {
    return nullptr;
  }

  auto it = histograms_.find(key);
  return &it->second;
}

double RayModel::SampleRange(const ObjectState &os, double sensor_theta, double sensor_phi) const {
  // Hacky, spoof a hit and make key
  double x = cos(sensor_phi) * cos(sensor_theta);
  double y = cos(sensor_phi) * sin(sensor_theta);
  double z = sin(sensor_phi);

  Observation obs(Eigen::Vector3d(x, y, z));
  ModelObservation mo(os, obs);

  auto *h = GetHistogram(mo);
  if (h == nullptr) {
    return 100.0;
  }

  // Sample range in model frame
  double dist_obs = h->Sample();

  double dist_obs_2d = dist_obs * obs.cos_phi;
  double range_xy = dist_obs_2d + obs.cos_theta * os.pos.x() + obs.sin_theta * os.pos.y();
  double range = range_xy / obs.cos_phi;

  if (range < 0) {
    return -1.0;
  }

  return range;
}

std::map<std::pair<double, double>, double> RayModel::GetHistogramFillinByAngle() const {
  std::map<std::pair<double, double>, double> counts;

 int n_angle = std::round(M_PI / kAngleRes) + 1;
  for (int i_theta = -n_angle; i_theta <= n_angle; i_theta++) {
    for (int i_phi = -n_angle; i_phi <= n_angle; i_phi++) {
      double theta = i_theta * kAngleRes;
      double phi = i_phi * kAngleRes;
      counts[std::pair<double, double>(theta, phi)] = 0;
    }
  }

  for (auto it = histograms_.begin(); it != histograms_.end(); it++) {
    const Histogram1GramKey &key = it->first;

    double theta = key.idx_theta * kAngleRes;
    double phi = key.idx_phi * kAngleRes;
    counts[std::pair<double, double>(theta, phi)]++;
  }

  int n_xy = 2*std::ceil(max_size_xy_ / kDistRes);
  int n_z = 2*std::ceil(max_size_z_ / kDistRes);
  int n_denom = n_xy * n_z;

  for (auto it = counts.begin(); it != counts.end(); it++) {
    it->second /= n_denom;
  }

  return counts;
}

void RayModel::PrintStats() const {
  printf("\tHave %ld histograms\n", histograms_.size());
}

void RayModel::Blur() {
  // Blur filter
  const double theta_std = ut::DegreesToRadians(5.0);
  const double phi_std = ut::DegreesToRadians(5.0);
  const double dist_ray_std = 0.10;
  const double dist_z_std = 0.10;
  const double std_dim[4] = {theta_std, phi_std, dist_ray_std, dist_z_std};
  const int range_std = 3;

  double res_dim[4] = {kAngleRes, kAngleRes, kDistRes, kDistRes};

  // Make lookup table for weights
  library::timer::Timer t;
  std::map< std::tuple<int, int, int, int>, double> weights;

  int n_dim[4] = {0, 0, 0, 0};

  for (size_t dim = 0; dim < 4; dim++) {
    n_dim[dim] = std::floor(std_dim[dim] * range_std / res_dim[dim]);

    for (int xd = -n_dim[dim]; xd <= n_dim[dim]; xd++) {
      int d_key[4] = {0, 0, 0, 0};
      d_key[dim] = xd;

      double dist = xd * res_dim[dim];
      double mahal_dist = dist / ( std_dim[dim] * std_dim[dim] );

      auto weights_key = std::make_tuple(d_key[0], d_key[1], d_key[2], d_key[3]);

      // Gaussian filter
      weights[weights_key] = exp(-0.5*mahal_dist);
    }
  }
  //printf("\tTook %5.3f sec to make %ld weight lookup table\n", t.GetSeconds(), weights.size());

  // Separable filter, blur each dimension individually
  std::map<Histogram1GramKey, library::histogram::Histogram> blurred_histograms;
  std::map<Histogram1GramKey, library::histogram::Histogram> prev_histograms(histograms_);

  // Go through each dim
  for (size_t dim = 0; dim < 4; dim++) {
    // Go through each element of previous blurred histogram result
    for (auto ph_it = prev_histograms.cbegin(); ph_it != prev_histograms.cend(); ph_it++) {
      const auto &key = ph_it->first;
      const auto &hist = ph_it->second;

      // Only blur along the dim we're considering
      for (int xd = -n_dim[dim]; xd <= n_dim[dim]; xd++) {
        int d_key[4] = {0, 0, 0, 0};
        d_key[dim] = xd;

        Histogram1GramKey blurred_key(key.idx_theta    + d_key[0],
                                      key.idx_phi      + d_key[1],
                                      key.idx_dist_ray + d_key[2],
                                      key.idx_dist_z   + d_key[3],
                                      kDistRes, kAngleRes, max_size_xy_, max_size_z_);

        // If we're out of range, don't blur
        if (!blurred_key.InRange()) {
          continue;
        }

        // Get histogram to add to
        auto blurred_histograms_it = blurred_histograms.find(blurred_key);

        // Create histogram if it's not there
        if (blurred_histograms_it == blurred_histograms.end()) {
          library::histogram::Histogram blurred_hist(-max_size_xy_, max_size_xy_, kDistRes);
          blurred_histograms.insert( std::pair<Histogram1GramKey, library::histogram::Histogram>(blurred_key, blurred_hist) );

          blurred_histograms_it = blurred_histograms.find(blurred_key);
          BOOST_ASSERT(blurred_histograms_it != blurred_histograms.end());
        }

        auto weights_key = std::make_tuple(d_key[0], d_key[1], d_key[2], d_key[3]);
        auto weights_it = weights.find(weights_key);
        BOOST_ASSERT(weights_it != weights.end());

        blurred_histograms_it->second.Add(hist, weights_it->second);
      }
    }

    prev_histograms = blurred_histograms;
    blurred_histograms.clear();
  }

  // Now we're done, save result
  histograms_ = prev_histograms;
}

} // namespace kitti
} // namespace app
