#include "library/sim_world/shape.h"

#include <Eigen/Geometry>

namespace library {
namespace sim_world {

Shape::Shape(const std::vector<Eigen::Vector2d> &corners, const std::string &name) :
  corners_(corners), center_(0, 0), name_(name) {
  for (const Eigen::Vector2d &c : corners_) {
    center_ += c;
  }
  center_ /= corners_.size();
}

double Shape::GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit) const {
  Eigen::Vector2d ray(cos(angle), sin(angle));
  return GetHit(origin, ray, hit);
}

double Shape::GetHit(const Eigen::Vector2d &origin, const Eigen::Vector2d &ray, Eigen::Vector2d *hit) const {

  Eigen::Vector2d ray_hat = ray.normalized();

  // Find line representing ray
  Eigen::Vector2d p_o1 = origin + ray_hat;

  Eigen::Vector3d originh = origin.homogeneous();
  Eigen::Vector3d p_o1h = p_o1.homogeneous();
  Eigen::Vector3d l_ray = originh.cross(p_o1h);

  bool valid = false;
  double best_distance = -1.0;

  for (size_t i=0; i<corners_.size(); i++) {
    // Find line representing edge of shape
    Eigen::Vector2d p_b0 = corners_[i];
    Eigen::Vector2d p_b1 = corners_[(i+1)%corners_.size()];

    // Check for line parallel to ray
    Eigen::Vector2d v_edge_hat = (p_b1 - p_b0).normalized();
    if (std::abs(ray_hat.dot(v_edge_hat)) > 1 - 1e-99) {
      //printf("\tparallel\n");
      continue;
    }

    Eigen::Vector3d p_b0h = p_b0.homogeneous();
    Eigen::Vector3d p_b1h = p_b1.homogeneous();
    Eigen::Vector3d l_edge = p_b0h.cross(p_b1h);

    // Find their intersection point
    Eigen::Vector3d cp = l_ray.cross(l_edge);

    // Check for intersection at infinity
    if (std::abs(cp(2)) < 1e-99) {
      //printf("\tintersect at inf: %5.3f\n", cp(2));
      continue;
    }

    Eigen::Vector2d p_intersect = cp.hnormalized();

    // Check for intersection in front of ray
    double distance = (p_intersect - origin).dot(ray_hat);
    if (distance <= 0.0f) {
      //printf("\tbehind\n");
      continue;
    }

    // Check for intersection on shape and in range
    Eigen::Vector2d p_b10 = p_b1 - p_b0;
    double x = (p_intersect - p_b0).dot(p_b10);
    if (x<=0.0f || x>=p_b10.dot(p_b10)) {
      //printf("\tnot on shape\n");
      continue;
    }

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

bool Shape::IsInside(double x, double y) const {
  // Count intersections
  Eigen::Vector2d origin(x, y);
  Eigen::Vector2d ray(0, 1);

  int count = 0;

  while (true) {
    Eigen::Vector2d hit;
    double distance = GetHit(origin, ray, &hit);

    if (distance < 0)
      break;

    //printf("intersect at %5.3f, %5.3f\n", hit(0), hit(1));

    count++;
    origin = hit + ray*1e-9;
  }

  //printf("count: %d\n", count);

  return (count%2) == 1;
}

bool Shape::Intersects(const Shape &shape) const {
  auto c = GetCenter();
  if (shape.IsInside(c(0), c(1))) {
    return true;
  }

  c = shape.GetCenter();
  if (IsInside(c(0), c(1))) {
    return true;
  }

  for (const auto& c : corners_) {
    if (shape.IsInside(c(0), c(1))) {
      return true;
    }
  }

  for (const auto& c : shape.corners_) {
    if (IsInside(c(0), c(1))) {
      return true;
    }
  }

  for (size_t i=0; i<corners_.size(); i++) {
    auto c0 = corners_[i];
    auto c1 = corners_[(i+1)%(corners_.size())];

    Eigen::Vector2d hit;

    double dist = shape.GetHit(c0, (c1 - c0), &hit);

    if (dist > 0.0 && dist < (c1 - c0).norm()) {
      return true;
    }
  }

  return false;
}

Eigen::Vector2d Shape::GetCenter() const {
  return center_;
}

double Shape::GetAngle() const {
  return angle_;
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

void Shape::Rotate(double angle_radians) {
  double s = sin(angle_radians);
  double c = cos(angle_radians);
  for (auto& corner : corners_) {
    auto c_center = corner - center_;
    double x = c_center(0)*c - c_center(1)*s;
    double y = c_center(0)*s + c_center(1)*c;

    corner(0) = x + center_(0);
    corner(1) = y + center_(1);
  }

  angle_ += angle_radians;
}

const std::string& Shape::GetName() const {
  return name_;
}

Shape Shape::CreateBox(double c_x, double c_y, double width, double length) {
  std::vector<Eigen::Vector2d> corners;

  corners.push_back(Eigen::Vector2d(c_x + length/2.0, c_y + width/2.0));
  corners.push_back(Eigen::Vector2d(c_x + length/2.0, c_y - width/2.0));
  corners.push_back(Eigen::Vector2d(c_x - length/2.0, c_y - width/2.0));
  corners.push_back(Eigen::Vector2d(c_x - length/2.0, c_y + width/2.0));

  return Shape(corners, "BOX");
}

Shape Shape::CreateStar(double c_x, double c_y, double size) {
  std::vector<Eigen::Vector2d> corners;

  for (int i=0; i<10; i++) {
    double r = (i%2) ? 1:0.5;
    r *= size;

    double angle = i * 36.0 * M_PI/180.0;
    corners.push_back(Eigen::Vector2d(c_x + r*sin(angle), c_y + r*cos(angle)));

  }

  return Shape(corners, "STAR");
}

}
}
