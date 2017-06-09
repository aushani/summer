#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<double> occupancies) {

  auto tic = std::chrono::steady_clock::now();

  std::uniform_int_distribution<> random_idx(0, points.size()-1);
  std::default_random_engine re;

  // Learn w
  w_.resize(n_inducing_points, 1);
  w_.setZero();

  int epochs = 1;
  int iterations = epochs*points.size();
  for (int i=0; i<iterations; i++) {
    if (i % (iterations/100) == 0) {
      printf(" %5.3f%% done\n", (100.0*i) / iterations);
    }

    int idx = random_idx(re);

    Point p = points[idx];
    double y = occupancies[idx];
    update_w(p, y);
  }
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Done learning w in %ld ms\n", t_ms.count());

}

double HilbertMap::k_sparse(Point p1, Point p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;

  double d2 = dx*dx + dy*dy;

  if (d2 > 1.0)
    return 0;

  double r = sqrt(d2);

  double t = 2 * M_PI * r;

  return (2 + cos(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sin(t);
}

Eigen::SparseMatrix<double> HilbertMap::phi_sparse(Point p) {
  Eigen::SparseMatrix<double> res;
  res.resize(n_inducing_points, 1);
  std::vector<int> support_idx = get_inducing_points_with_support(p);
  for (int idx : support_idx) {
    Point x_m = get_inducing_point(idx);
    double k = k_sparse(p, x_m);
    res.coeffRef(idx, 0) = k;
  }

  return res;
}

Eigen::SparseMatrix<double> HilbertMap::gradient(Point x, double y) {
  Eigen::SparseMatrix<double> phi = phi_sparse(x);
  auto wTphi = w_.transpose() * phi;
  double c = -y * 1.0/(1.0 + expf(y * wTphi(0, 0)));

  Eigen::SparseMatrix<double> grad = c * phi;

  // Add regularization gradient
  // TODO
  //double lambda_1 = 0.001;
  //double lambda_2 = 0.150;
  //for (int i=0; i<w.size(); i++) {
  //  int sgn_w = (w[i] > 0) - (w[i] < 0);
  //  grad[i] += lambda_1*2*w[i] + lambda_2*sgn_w;
  //}

  return grad;
}

void HilbertMap::update_w(Point x, double y) {
  double learning_rate = 0.1;

  Eigen::SparseMatrix<double> grad = gradient(x, y);
  w_ -= learning_rate * grad;
}

double HilbertMap::get_occupancy(Point p) {
  Eigen::SparseMatrix<double> phi = phi_sparse(p);
  auto wTphi = w_.transpose() * phi;
  return 1.0 / (1.0 + expf(wTphi(0, 0)));
}

Point HilbertMap::get_inducing_point(int idx) {
  double min = -25.0;
  double max =  25.0;

  int i = idx / inducing_points_n_dim;
  int j = idx % inducing_points_n_dim;

  double x = min + i * (max - min) / inducing_points_n_dim;
  double y = min + j * (max - min) / inducing_points_n_dim;

  return Point(x, y);
}

std::vector<int> HilbertMap::get_inducing_points_with_support(Point p) {
  double min = -25.0;
  double max =  25.0;

  double step = (max - min)/inducing_points_n_dim;

  int i0 = (p.x - min) / step + 0.5;
  int j0 = (p.y - min) / step + 0.5;

  int idx_step = 1.0 / step;
  idx_step++;

  std::vector<int> res;

  for (int di = -idx_step; di<=idx_step; di++) {
    for (int dj = -idx_step; dj<=idx_step; dj++) {
      int i = i0 + di;
      int j = j0 + dj;
      int idx = i*inducing_points_n_dim + j;

      res.push_back(idx);
    }
  }

  return res;
}
