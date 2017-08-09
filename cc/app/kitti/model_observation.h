#pragma once

#include "library/util/angle.h"

#include "app/kitti/observation.h"
#include "app/kitti/object_state.h"

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

};

} // namespace kitti
} // namespace app
