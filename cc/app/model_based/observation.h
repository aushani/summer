#pragma once

#include <Eigen/Core>

// assume origin at (0, 0) for now
class Observation {
 public:
  explicit Observation(const Eigen::Vector2d &x);
  explicit Observation(double range, double theta);

  double GetRange() const;
  double GetTheta() const;

  double GetCosTheta() const;
  double GetSinTheta() const;

  double GetX() const;
  double GetY() const;
  const Eigen::Vector2d& GetPos() const;

 private:
  Eigen::Vector2d pos_;
  double range_;
  double theta_;

  double cos_theta_;
  double sin_theta_;
};
