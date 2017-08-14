#pragma once

#include <Eigen/Core>

namespace app {
namespace kitti {

class ObjectState {
 public:
  ObjectState(const Eigen::Vector2d &x, double t, const std::string &cn) :
    pos(x), theta(t), classname(cn) {
  }

  bool operator<(const ObjectState &os) const {
    for (int i=0; i<3; i++) {
      if (std::abs(pos(i) - os.pos(i)) > kResPos_) {
        return pos(i) < os.pos(i);
      }
    }

    if (std::abs(theta - os.theta) > kResAngle_) {
      return theta < os.theta;
    }

    return classname < os.classname;
  }

  const Eigen::Vector3d pos;
  const double theta;

  const std::string classname;

 private:
  double kResPos_ = 0.001;
  double kResAngle_ = 0.001;

};

} // namespace kitti
} // namespace app
