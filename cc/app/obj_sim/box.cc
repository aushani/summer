#include "box.h"

#include <Eigen/Geometry>

Box::Box(double c_x, double c_y, double width, double length) :
  c_x_(c_x),
  c_y_(c_y) {

  // Box corners
  corners_.push_back(Eigen::Vector2d(c_x + length/2.0, c_y + width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x + length/2.0, c_y - width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x - length/2.0, c_y - width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x - length/2.0, c_y + width/2.0));

  // Make a star for shits
  //for (int i=0; i<10; i++) {
  //  double angle = 36 * i * M_PI/180.0;
  //  double radius = (i%2) ? 1:0.5;
  //  double x = c_x + width*radius*sin(angle);
  //  double y = c_y + length*radius*cos(angle);
  //  corners_.push_back(Eigen::Vector2d(x, y));
  //}
}

double Box::GetHit(const Eigen::Vector2d &origin, double angle, Eigen::Vector2d *hit) {

  // Find line representing ray
  Eigen::Vector2d ray(cos(angle), sin(angle));
  Eigen::Vector2d p_o1 = origin + ray;

  Eigen::Vector3d originh = origin.homogeneous();
  Eigen::Vector3d p_o1h = p_o1.homogeneous();
  Eigen::Vector3d l_ray = originh.cross(p_o1h);

  bool valid = false;
  double best_distance = -1.0;

  for (size_t i=0; i<corners_.size(); i++) {
    // Find line representing edge of box
    Eigen::Vector2d p_b0 = corners_[i];
    Eigen::Vector2d p_b1 = corners_[(i+1)%corners_.size()];

    Eigen::Vector3d p_b0h = p_b0.homogeneous();
    Eigen::Vector3d p_b1h = p_b1.homogeneous();
    Eigen::Vector3d l_edge = p_b0h.cross(p_b1h);

    // Find their intersection point
    Eigen::Vector2d p_intersect = l_ray.cross(l_edge).hnormalized();

    // Check for intersection in front of ray
    double distance = (p_intersect - origin).dot(ray);
    if (distance < 0)
      continue;

    // Check for intersection on box and in range
    Eigen::Vector2d p_b10 = p_b1 - p_b0;
    double x = (p_intersect - p_b0).dot(p_b10);
    if (x<0.0 || x>p_b10.dot(p_b10))
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

bool Box::IsInside(double x, double y) {

  Eigen::Vector2d point(x, y);

  // Sum up angles
  double angle = 0.0;
  for (int i=0; i<4; i++) {
    Eigen::Vector2d c0 = corners_[i];
    Eigen::Vector2d c1 = corners_[(i+1)%4];

    Eigen::Vector2d pc0 = (c0 - point).normalized();
    Eigen::Vector2d pc1 = (c1 - point).normalized();

    angle += acos(pc0.dot(pc1)) * 180.0/M_PI;
  }

  if (angle >= 359.9) { // tol
    return true;
  }

  return false;
}

/*
bool Box::IsInside(double x, double y) {

  Eigen::Vector2d point(x, y);

  // Sum up angles
  double total_angle = 0.0;
  for (size_t i=0; i<corners_.size(); i++) {
    Eigen::Vector2d c0 = corners_[i];
    Eigen::Vector2d c1 = corners_[(i+1)%corners_.size()];

    Eigen::Vector2d pc0 = (c0 - point).normalized();
    Eigen::Vector2d pc1 = (c1 - point).normalized();
    double cos_angle = pc0.dot(pc1);

    Eigen::Vector3d pc0h(pc0(0), pc0(1), 0);
    Eigen::Vector3d pc1h(pc1(0), pc1(1), 0);
    Eigen::Vector3d cross = pc0h.cross(pc1h);
    double sin_angle = cross(2);

    double angle = atan2(sin_angle, cos_angle) * 180.0/M_PI;
    total_angle += angle;
  }

  int winding_num = round(total_angle/360.0);

  return (winding_num % 2) == 0;
}
*/

double Box::GetCenterX() const {
  return c_x_;
}

double Box::GetCenterY() const {
  return c_y_;
}
