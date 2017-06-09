#pragma once

#include <vector>

struct Point {
  double x;
  double y;

  Point(double xx, double yy) : x(xx), y(yy) {;}
};

class HilbertMap {
 public:
  HilbertMap(std::vector<Point> points, std::vector<double> occupancies);

  double get_occupancy(Point p);

 private:
  std::vector<Point> inducing_points_;
  std::vector<double> w_;

  double k_sparse(Point p1, Point p2);
  std::vector<double> phi_sparse(Point p);
  std::vector<double> gradient(Point x, double y);
  void update_w(Point x, double y);
};
