#pragma once

#include <Eigen/Core>

namespace app {
namespace model_based {

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
  double kResPos_ = 0.001;
  double kResAngle_ = 0.001;

  Eigen::Vector2d pos_;
  double theta_;
  std::string classname_;

  double bearing_;

  double range_;
  double cos_theta_;
  double sin_theta_;
};

} // namespace model_based
} // namespace app
