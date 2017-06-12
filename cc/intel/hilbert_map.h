#pragma once

#include <vector>
#include <string>

struct Point {
  float x;
  float y;

  Point() : x(0.0f), y(0.0f) {;}
  Point(float xx, float yy) : x(xx), y(yy) {;}
};

class HilbertMap {
 public:
  HilbertMap(std::vector<Point> points, std::vector<float> occupancies);
  HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  float get_occupancy(Point p);

 private:
  const int inducing_points_n_dim = 100;
  const int n_inducing_points = inducing_points_n_dim*inducing_points_n_dim;

  float *d_w_ = NULL;

  std::vector<Point> inducing_points_;

  void update_w(Point x, float y);

  Point get_inducing_point(int idx);

  std::vector<int> get_inducing_points_with_support(Point p);
};
