#pragma once

#include <vector>
#include <Eigen/Core>

class Box {
 public:
  Box(double c_x, double c_y, double width, double length);

  double GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit);

 private:
  std::vector<Eigen::Vector2d> corners_;
};
