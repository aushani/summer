#include "object_state.h"

ObjectState::ObjectState(double x, double y, double a, const std::string &cn) :
 pos_(x, y), theta_(a), classname_(cn),
 bearing_(atan2(y, x)), range_(pos_.norm()), cos_theta_(cos(theta_)), sin_theta_(sin(theta_)) {

  if (GetRange() > max_size_) {
    max_dtheta_ = asin((max_size_ + 2 * kDistanceStep_) / GetRange());
  }
}

const Eigen::Vector2d& ObjectState::GetPos() const {
  return pos_;
}

double ObjectState::GetRange() const {
  return range_;
}

double ObjectState::GetTheta() const {
  return theta_;
}

double ObjectState::GetCosTheta() const {
  return cos_theta_;
}

double ObjectState::GetSinTheta() const {
  return sin_theta_;
}

const std::string& ObjectState::GetClassname() const {
  return classname_;
}

double ObjectState::GetBearing() const {
  return bearing_;
}

double ObjectState::GetMaxDtheta() const {
  return max_dtheta_;
}

bool ObjectState::operator<(const ObjectState &os) const {
  if (std::abs(pos_(0) - os.pos_(0)) > kResPos_)
    return pos_(0) < os.pos_(0);

  if (std::abs(pos_(1) - os.pos_(1)) > kResPos_)
    return pos_(1) < os.pos_(1);

  if (std::abs(theta_ - os.theta_) > kResAngle_)
    return theta_ < os.theta_;

  return classname_ < os.classname_;
}
