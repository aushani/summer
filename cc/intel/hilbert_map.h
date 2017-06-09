#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

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
  const int inducing_points_n_dim = 100;
  const int n_inducing_points = inducing_points_n_dim*inducing_points_n_dim;

  Eigen::MatrixXd w_;

  std::vector<Point> inducing_points_;

  double k_sparse(Point p1, Point p2);
  Eigen::SparseMatrix<double> phi_sparse(Point p);
  Eigen::SparseMatrix<double> gradient(Point x, double y);
  void update_w(Point x, double y);

  Point get_inducing_point(int idx);

  std::vector<int> get_inducing_points_with_support(Point p);
};
