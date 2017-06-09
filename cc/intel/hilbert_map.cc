#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<double> occupancies) {

  auto tic = std::chrono::steady_clock::now();

  std::uniform_int_distribution<> random_idx(0, points.size()-1);
  std::default_random_engine re;

  // Make inducing points
  /*
  for (int i=0; i<2000; i++) {
    int idx = random_idx(re);
    inducing_points_.push_back(points[idx]);
  }
  */
  double x_min = -12;
  double x_max = 20;

  double y_min = -25;
  double y_max = 8;
  int n_dim = 100;
  for (int i=0; i<n_dim; i++) {
    double x = x_min + i*(x_max - x_min)/n_dim;
    for (int j=0; j<n_dim; j++) {
      double y = y_min + j*(y_max - y_min)/n_dim;

      inducing_points_.push_back(Point(x, y));
    }
  }

  // Learn w
  for (size_t i=0; i<inducing_points_.size(); i++) {
    w_.push_back(0.0);
  }

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

std::vector<double> HilbertMap::phi_sparse(Point p) {
  std::vector<double> res;
  for ( Point &x_m : inducing_points_) {
    res.push_back(k_sparse(p, x_m));
  }

  return res;
}

std::vector<double> HilbertMap::gradient(Point x, double y) {
  std::vector<double> grad;

  std::vector<double> phi = phi_sparse(x);
  double wTphi = 0.0;
  for (size_t i=0; i<w_.size(); i++) {
    wTphi += w_[i]*phi[i];
  }
  double c = -y * 1.0/(1.0 + expf(y * wTphi));

  for (size_t i=0; i<w_.size(); i++) {
    grad.push_back(c*phi[i]);
  }

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

  std::vector<double> grad = gradient(x, y);
  for (size_t i=0; i<w_.size(); i++) {
    w_[i] -= learning_rate * grad[i];
  }
}

double HilbertMap::get_occupancy(Point p) {
  std::vector<double> phi = phi_sparse(p);
  double wTphi = 0.0;
  for (size_t i = 0; i < w_.size(); i++) {
    wTphi += w_[i] * phi[i];
  }
  return 1.0 / (1.0 + expf(wTphi));
}
