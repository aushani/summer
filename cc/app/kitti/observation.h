#pragma once

#include <Eigen/Core>

namespace app {
namespace kitti {

// assume origin at (0, 0, 0) for now
struct Observation {
  explicit Observation(const Eigen::Vector3d &x) :
   pos(x),
   range(pos.norm()),
   range_xy(sqrt(pos(0)*pos(0) + pos(1)*pos(1))),
   theta(atan2(pos(1), pos(0))),
   phi(atan2(pos(2), range_xy)),
   cos_theta(cos(theta)),
   sin_theta(sin(theta)),
   cos_phi(cos(phi)),
   sin_phi(sin(phi)) { };

  const Eigen::Vector3d pos;

  const double range;
  const double range_xy;

  const double theta;
  const double phi;

  const double cos_theta;
  const double sin_theta;

  const double cos_phi;
  const double sin_phi;
};

} // namespace kitti
} // namespace app
