#include "observation.h"

Observation::Observation(const Eigen::Vector2d &x) :
 pos_(x),
 range_(pos_.norm()),
 theta_(atan2(pos_(1), pos_(0))),
 cos_theta_(cos(theta_)),
 sin_theta_(sin(theta_)) {
}

Observation::Observation(double range, double theta) :
 pos_(range*cos(theta), range*sin(theta)),
 range_(range),
 theta_(theta),
 cos_theta_(cos(theta_)),
 sin_theta_(sin(theta_)) {
  while (theta < -M_PI) theta += 2*M_PI;
  while (theta >  M_PI) theta -= 2*M_PI;
}

double Observation::GetRange() const {
  return range_;
}

double Observation::GetTheta() const {
  return theta_;
}

double Observation::GetCosTheta() const {
  return cos_theta_;
}

double Observation::GetSinTheta() const {
  return sin_theta_;
}

double Observation::GetX() const {
  return pos_(0);
}

double Observation::GetY() const {
  return pos_(1);
}

const Eigen::Vector2d& Observation::GetPos() const {
  return pos_;
}
