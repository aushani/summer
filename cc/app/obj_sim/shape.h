#pragma once

#include <vector>
#include <Eigen/Core>

class Shape {
 public:
  Shape(const std::vector<Eigen::Vector2d> &corners);

  double GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit);
  double GetHit(const Eigen::Vector2d &origin, const Eigen::Vector2d &ray, Eigen::Vector2d *hit);

  bool IsInside(double x, double y);

  Eigen::Vector2d GetCenter() const;

  double GetMinX() const;
  double GetMaxX() const;
  double GetMinY() const;
  double GetMaxY() const;

  static Shape CreateBox(double c_x, double c_y, double width, double length);
  static Shape CreateStar(double c_x, double c_y, double size);

 private:
  std::vector<Eigen::Vector2d> corners_;
};
