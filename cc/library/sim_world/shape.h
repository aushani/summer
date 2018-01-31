#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>

namespace library {
namespace sim_world {

class Shape {
 public:
  Shape(const std::vector<Eigen::Vector2d> &corners, const std::string &name);

  double GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit) const;
  double GetHit(const Eigen::Vector2d &origin, const Eigen::Vector2d &ray, Eigen::Vector2d *hit) const;

  bool IsInside(double x, double y) const;
  bool Intersects(const Shape &shape) const;

  const Eigen::Vector2d& GetCenter() const;
  double GetAngle() const;

  double GetMinX() const;
  double GetMaxX() const;
  double GetMinY() const;
  double GetMaxY() const;

  void Rotate(double angle_radians);

  const std::string& GetName() const;

  static Shape CreateBox(double c_x, double c_y, double width, double length);
  static Shape CreateStar(double c_x, double c_y, double size);

 private:
  std::vector<Eigen::Vector2d> corners_;
  Eigen::Vector2d center_;
  double angle_ = 0.0;

  std::string name_;
};

}
}
