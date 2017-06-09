#pragma once

#include <vector>
#include <string>

struct Point {
  double x;
  double y;

  Point(double xx, double yy) : x(xx), y(yy) {;}
};

class HilbertMap {
 public:
  HilbertMap(std::vector<Point> points, std::vector<double> occupancies);
  HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  double get_occupancy(Point p);

 private:
  const int inducing_points_n_dim = 100;
  const int n_inducing_points = inducing_points_n_dim*inducing_points_n_dim;

  float *w_ = NULL;

  std::vector<Point> inducing_points_;

  double k_sparse(const Point &p1, const Point &p2);
  void update_w(Point x, double y);

  Point get_inducing_point(int idx);

  std::vector<int> get_inducing_points_with_support(Point p);
};
