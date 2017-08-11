#include "app/kitti/ray_model.h"

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

std::map<std::pair<double, double>, int> RayModel::GetHistogramCountsByAngle() const {
  std::map<std::pair<double, double>, int> counts;

  for (auto it = histograms_.begin(); it != histograms_.end(); it++) {
    const Histogram1GramKey &key = it->first;

    double theta = key.idx_theta * kAngleRes;
    double phi = key.idx_phi * kAngleRes;
    counts[std::pair<double, double>(theta, phi)]++;
  }

  return counts;
}

void RayModel::PrintStats() const {
  printf("\tHave %ld histograms\n", histograms_.size());
}

void RayModel::Blur() {
  std::map<Histogram1GramKey, library::histogram::Histogram> blurred_histograms;

  double theta_std = ut::DegreesToRadians(5.0);
  double phi_std = ut::DegreesToRadians(5.0);
  double dist_ray_std = 0.10;
  double dist_z_std = 0.10;
  int range_std = 3;

  Eigen::Matrix<double, 4, 4> sigma;
  sigma.setZero();
  sigma.diagonal()[0] = theta_std*theta_std;
  sigma.diagonal()[1] = phi_std*phi_std;
  sigma.diagonal()[2] = dist_ray_std*dist_ray_std;
  sigma.diagonal()[3] = dist_z_std*dist_z_std;

  auto sigma_inv = sigma.inverse();

  // Make lookup table for weights
  std::map< std::tuple<int, int, int, int>, double> weights;
  int n_theta = std::floor( theta_std     * range_std / kAngleRes );
  int n_phi   = std::floor( phi_std       * range_std / kAngleRes );
  int n_dr    = std::floor( dist_ray_std  * range_std / kDistRes  );
  int n_dz    = std::floor( dist_z_std    * range_std / kDistRes  );

  for (int x1 = -n_theta; x1 <= n_theta; x1++) {
    double dtheta = x1 * kAngleRes;

    for (int x2 = -n_phi; x2 <= n_phi; x2++) {
      double dphi = x2 * kAngleRes;

      for (int x3 = -n_dr; x3 <= n_dr; x3++) {
        double ddr = x3 * kDistRes;

        for (int x4 = -n_dz; x4 <= n_dz; x4++) {
          double ddz = x4 * kDistRes;

          Eigen::Vector4d d_key(dtheta, dphi, ddr, ddz);
          double dist = d_key.transpose() * sigma_inv * d_key;

          weights[std::tie(x1, x2, x3, x4)] = exp(-dist);
        }
      }
    }
  }

  size_t count = 0;
  library::timer::Timer t_total;
  library::timer::Timer t_step;

  for (auto it = histograms_.cbegin(); it != histograms_.cend(); it++) {
    if (t_step.GetSeconds() > 10) {
      printf("\tProcessing histogram %ld / %ld, %5.3f sec / histogram\n",
          count + 1, histograms_.size(), t_total.GetSeconds() / count);
      t_step.Start();
    }

    const auto &key = it->first;
    const auto &hist = it->second;

    for (int x1 = -n_theta; x1 <= n_theta; x1++) {
      for (int x2 = -n_phi; x2 <= n_phi; x2++) {
        for (int x3 = -n_dr; x3 <= n_dr; x3++) {
          for (int x4 = -n_dz; x4 <= n_dz; x4++) {

            Histogram1GramKey blurred_key(key.idx_theta    + x1,
                                          key.idx_phi      + x2,
                                          key.idx_dist_ray + x3,
                                          key.idx_dist_z   + x4,
                                          kDistRes, kAngleRes, max_size_xy_, max_size_z_);

            if (!blurred_key.InRange()) {
              continue;
            }

            // Create histogram if it's not there
            if (blurred_histograms.count(key) == 0) {
              library::histogram::Histogram blurred_hist(-max_size_xy_, max_size_xy_, kDistRes);
              blurred_histograms[key] = blurred_hist;
            }

            double weight = weights[std::tie(x1, x2, x3, x4)];
            blurred_histograms[key].Add(hist, weight);
          }
        }
      }
    }

    count++;
  }

  histograms_ = blurred_histograms;
}

} // namespace kitti
} // namespace app
