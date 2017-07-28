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

  double GetMaxDtheta() const;

  bool operator<(const ObjectState &os) const;

 private:
  double kResPos_ = 0.001;
  double kResAngle_ = 0.001;

  double max_size_ = 5.0;
  double kDistanceStep_ = 0.15; // 15 cm

  Eigen::Vector2d pos_;
  double theta_;
  std::string classname_;

  double bearing_;

  double range_;
  double cos_theta_;
  double sin_theta_;

  double max_dtheta_ = 2*M_PI;
};
