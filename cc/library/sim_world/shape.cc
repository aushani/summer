#include "library/sim_world/shape.h"

#include <Eigen/Geometry>

namespace library {
namespace sim_world {

Shape::Shape(const std::vector<Eigen::Vector2d> &corners) :
  corners_(corners) {
}

double Shape::GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit) const {
  Eigen::Vector2d ray(cos(angle), sin(angle));
  return GetHit(origin, ray, hit);
}

double Shape::GetHit(const Eigen::Vector2d &origin, const Eigen::Vector2d &ray, Eigen::Vector2d *hit) const {

  // Find line representing ray
  Eigen::Vector2d p_o1 = origin + ray;

  Eigen::Vector3d originh = origin.homogeneous();
  Eigen::Vector3d p_o1h = p_o1.homogeneous();
  Eigen::Vector3d l_ray = originh.cross(p_o1h);

  bool valid = false;
  double best_distance = -1.0;

  for (size_t i=0; i<corners_.size(); i++) {
    // Find line representing edge of shape
    Eigen::Vector2d p_b0 = corners_[i];
    Eigen::Vector2d p_b1 = corners_[(i+1)%corners_.size()];

    Eigen::Vector3d p_b0h = p_b0.homogeneous();
    Eigen::Vector3d p_b1h = p_b1.homogeneous();
    Eigen::Vector3d l_edge = p_b0h.cross(p_b1h);

    // Find their intersection point
    Eigen::Vector2d p_intersect = l_ray.cross(l_edge).hnormalized();

    // Check for intersection in front of ray
    double distance = (p_intersect - origin).dot(ray);
    if (distance <= 0.0f)
      continue;

    // Check for intersection on shape and in range
    Eigen::Vector2d p_b10 = p_b1 - p_b0;
    double x = (p_intersect - p_b0).dot(p_b10);
    if (x<=0.0f || x>p_b10.dot(p_b10))
      continue;

    // Check to see if this is the closest hit for the ray
    if (valid && best_distance < distance)
      continue;

    // Update hit
    *hit = p_intersect;
    valid = true;
    best_distance = distance;
  }

  return best_distance;
}

bool Shape::IsInside(double x, double y) {

  // TODO Really should check being colinear
  // And handle literal corner cases better
  for (double angle = 0.4242; angle<2*M_PI; angle+=0.234) {
    // Count intersections
    Eigen::Vector2d origin(x, y);
    Eigen::Vector2d ray(cos(angle), sin(angle));

    int count = 0;

    while (true) {
      Eigen::Vector2d hit;
      double distance = GetHit(origin, ray, &hit);

      if (distance < 0)
        break;

      count++;
      origin = hit + ray*1e-5;
    }

    if (count%2 == 0)
      return false;
  }

  return true;
}

Eigen::Vector2d Shape::GetCenter() const {
  Eigen::Vector2d center(0, 0);

  for (const Eigen::Vector2d &c : corners_) {
    center += c;
  }
  center /= corners_.size();

  return center;
}

double Shape::GetMinX() const {
  double min_x = 0.0f;
  bool first = true;

  for (const Eigen::Vector2d &c : corners_) {
    double x = c(0);
    if ( x < min_x || first) {
      min_x = x;
    }
    first = false;
  }

  return min_x;
}

double Shape::GetMaxX() const {
  double max_x = 0.0f;
  bool first = true;

  for (const Eigen::Vector2d &c : corners_) {
    double x = c(0);
    if ( x > max_x || first) {
      max_x = x;
    }
    first = false;
  }

  return max_x;
}

double Shape::GetMinY() const {
  double min_y = 0.0f;
  bool first = true;

  for (const Eigen::Vector2d &c : corners_) {
    double y = c(1);
    if ( y < min_y || first) {
      min_y = y;
    }
    first = false;
  }

  return min_y;
}

double Shape::GetMaxY() const {
  double max_y = 0.0f;
  bool first = true;

  for (const Eigen::Vector2d &c : corners_) {
    double y = c(1);
    if ( y > max_y || first) {
      max_y = y;
    }
    first = false;
  }

  return max_y;
}

Shape Shape::CreateBox(double c_x, double c_y, double width, double length) {
  std::vector<Eigen::Vector2d> corners;

  corners.push_back(Eigen::Vector2d(c_x + length/2.0, c_y + width/2.0));
  corners.push_back(Eigen::Vector2d(c_x + length/2.0, c_y - width/2.0));
  corners.push_back(Eigen::Vector2d(c_x - length/2.0, c_y - width/2.0));
  corners.push_back(Eigen::Vector2d(c_x - length/2.0, c_y + width/2.0));

  return Shape(corners);
}

Shape Shape::CreateStar(double c_x, double c_y, double size) {
  std::vector<Eigen::Vector2d> corners;

  for (int i=0; i<10; i++) {
    double r = (i%2) ? 1:0.5;
    r *= size;

    double angle = i * 36.0 * M_PI/180.0;
    corners.push_back(Eigen::Vector2d(c_x + r*sin(angle), c_y + r*cos(angle)));

  }

  return Shape(corners);
}

}
}
