#pragma once

#include <Eigen/Core>

class ObjectState {
 public:
  ObjectState(double x, double y, double a, const std::string &cn);

  const Eigen::Vector2d& GetPos() const;
  double GetTheta() const;
  const std::string& GetClassname() const;

  double GetRange() const;
  double GetCosTheta() const;
  double GetSinTheta() const;

  double GetBearing() const;

  bool operator<(const ObjectState &os) const;

 private:
  const double kResPos_ = 0.001;
  const double kResAngle_ = 0.001;

  Eigen::Vector2d pos_;
  double theta_;
  std::string classname_;

  double bearing_;

  double range_;
  double cos_theta_;
  double sin_theta_;

};
