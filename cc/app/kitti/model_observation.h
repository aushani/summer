#pragma once

#include "library/kitti/velodyne_scan.h"
#include "library/util/angle.h"

#include "app/kitti/observation.h"
#include "app/kitti/object_state.h"

namespace kt = library::kitti;

namespace app {
namespace kitti {

struct ModelObservation {
  double theta;         // Heading of ray relative to object
  double phi;           // Pitch vertically of ray relative to object
  double dist_ray;      // Closest distance of ray to object center in xy plane
  double dist_z;        // Closest distance of ray to object center in z axis

  double dist_obs;      // Distance of range observation relative to closest point of ray to object center

  bool in_front;        // Check to make sure that the object is not behind us

  ModelObservation(const ObjectState &os, const Observation &x_hit) {
    theta = library::util::MinimizeAngle(x_hit.theta - os.theta);
    phi = library::util::MinimizeAngle(x_hit.phi);

    // in xy-plane
    dist_ray = x_hit.sin_theta*os.pos.x() - x_hit.cos_theta*os.pos.y();

    // in xy-plane
    double dist_obs_2d = x_hit.range_xy - x_hit.cos_theta*os.pos.x() - x_hit.sin_theta*os.pos.y();

    // account for z
    dist_obs = dist_obs_2d / x_hit.cos_phi;

    double dist_to_obj = os.pos.norm();
    double z_a = dist_to_obj * x_hit.sin_phi;
    dist_z = os.pos.z() - z_a;

    in_front = dist_obs < x_hit.range;
  }

  static std::vector<ModelObservation> MakeModelObservations(const ObjectState &os, const kt::VelodyneScan &scan, double max_size_xy, double max_size_z) {
    std::vector<Observation> obs;
    for (const auto &x_hit: scan.GetHits()) {
      obs.emplace_back(x_hit);
    }

    return MakeModelObservations(os, obs, max_size_xy, max_size_z);
  }

  static std::vector<ModelObservation> MakeModelObservations(const ObjectState &os, const std::vector<Observation> &observations,
      double max_size_xy, double max_size_z) {
    std::vector<ModelObservation> mos;

    double dist_to_obj = os.pos.norm();

    for (const auto &obs : observations) {
      ModelObservation mo;

      // in xy-plane
      mo.dist_ray = obs.sin_theta*os.pos.x() - obs.cos_theta*os.pos.y();

      // Check if ray comes close enough to object
      if (std::abs(mo.dist_ray) > max_size_xy) {
        continue;
      }

      // in xy-plane
      double dist_obs_2d = obs.range_xy - obs.cos_theta*os.pos.x() - obs.sin_theta*os.pos.y();

      // account for z
      mo.dist_obs = dist_obs_2d / obs.cos_phi;

      // Check for occlusion
      if (mo.dist_obs < -max_size_xy) {
        continue;
      }

      double z_a = dist_to_obj * obs.sin_phi;
      mo.dist_z = os.pos.z() - z_a;

      if (std::abs(mo.dist_z) > max_size_z) {
        continue;
      }

      mo.in_front = mo.dist_obs < obs.range;

      if (!mo.in_front) {
        continue;
      }

      mo.theta = library::util::MinimizeAngle(obs.theta - os.theta);
      mo.phi = library::util::MinimizeAngle(obs.phi);


      mos.push_back(mo);
    }

    return mos;
  }

 private:
  ModelObservation() {;}

};

} // namespace kitti
} // namespace app
