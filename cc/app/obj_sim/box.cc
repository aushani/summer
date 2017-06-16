#include "box.h"

#include <Eigen/Geometry>

Box::Box(double c_x, double c_y, double width, double length) {

  // Box corners
  corners_.push_back(Eigen::Vector2d(c_x + length/2.0, c_y + width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x + length/2.0, c_y - width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x - length/2.0, c_y - width/2.0));
  corners_.push_back(Eigen::Vector2d(c_x - length/2.0, c_y + width/2.0));
}

double Box::GetHit(const Point &origin, double angle, Point *hit) {

  // Find line representing ray
  Eigen::Vector2d p_o0(origin.x, origin.y);
  Eigen::Vector2d ray(cos(angle), sin(angle));
  Eigen::Vector2d p_o1 = p_o0 + ray;

  Eigen::Vector3d p_o0h = p_o0.homogeneous();
  Eigen::Vector3d p_o1h = p_o1.homogeneous();
  Eigen::Vector3d l_ray = p_o0h.cross(p_o1h);

  bool valid = false;
  double best_distance = -1.0;

  for (int i=0; i<4; i++) {
    // Find line representing edge of box
    Eigen::Vector2d p_b0 = corners_[i];
    Eigen::Vector2d p_b1 = corners_[(i+1)%4];

    Eigen::Vector3d p_b0h = p_b0.homogeneous();
    Eigen::Vector3d p_b1h = p_b1.homogeneous();
    Eigen::Vector3d l_edge = p_b0h.cross(p_b1h);

    // Find their intersection point
    Eigen::Vector2d p_intersect = l_ray.cross(l_edge).hnormalized();

    // Check for intersection in front of ray
    double distance = (p_intersect - p_o0).dot(ray);
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
    hit->x = p_intersect(0);
    hit->y = p_intersect(1);
    valid = true;
    best_distance = distance;
  }

  return best_distance;
}
