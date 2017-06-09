#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>
#include <map>
#include <cstring>

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<double> occupancies) {

  std::uniform_int_distribution<> random_idx(0, points.size()-1);
  std::default_random_engine re;

  // Learn w
  w_ = new float[n_inducing_points];
  memset(w_, 0, sizeof(float)*n_inducing_points);

  int epochs = 1;
  int iterations = epochs*points.size();
  //iterations = 10000;
  auto tic = std::chrono::steady_clock::now();
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

HilbertMap::HilbertMap(const HilbertMap &hm) : inducing_points_(hm.inducing_points_) {
  w_ = new float[n_inducing_points];
  memcpy(w_, hm.w_, sizeof(float)*n_inducing_points);
}

HilbertMap::~HilbertMap() {
  if (w_)
    delete[] w_;
}

double HilbertMap::k_sparse(const Point &p1, const Point &p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;

  double d2 = dx*dx + dy*dy;

  if (d2 > 1.0)
    return 0;

  double r = sqrt(d2);

  double t = 2 * M_PI * r;

  return (2 + cos(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sin(t);
}

void HilbertMap::update_w(Point x, double y) {
  double learning_rate = 0.1;

  //Eigen::SparseMatrix<double> grad = gradient(x, y);
  //w_ -= learning_rate * grad;

  // Where does this point actually have support?
  std::vector<int> support_idx = get_inducing_points_with_support(x);

  // Evaluate kernel
  std::map<int, double> phi_sparse;
  std::vector<int> nz;
  for (int i : support_idx) {
    double k = k_sparse(x, get_inducing_point(i));
    if (k>1e-5) {
      phi_sparse[i] = k;
      nz.push_back(i);
    }
  }

  // Compute gradient and update w
  double wTphi = 0.0;
  for (int i : nz) {
    wTphi += w_[i] * phi_sparse[i];
  }
  double c = -y * 1.0/(1.0 + expf(y * wTphi));

  // TODO no regularization

  for (int i : nz) {
    w_[i] -= learning_rate * c * phi_sparse[i];
  }
}

double HilbertMap::get_occupancy(Point p) {
  // Where does this point actually have support?
  std::vector<int> support_idx = get_inducing_points_with_support(p);

  // Evaluate
  double wTphi = 0.0;
  for (int i : support_idx) {
    wTphi += w_[i] * k_sparse(p, get_inducing_point(i));
  }
  return 1.0 / (1.0 + expf(wTphi));
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
    int i = i0 + di;
    if (i < 0 || i >= inducing_points_n_dim)
      continue;
    for (int dj = -idx_step; dj<=idx_step; dj++) {
      int j = j0 + dj;
      if (j < 0 || j >= inducing_points_n_dim)
        continue;
      int idx = i*inducing_points_n_dim + j;

      res.push_back(idx);
    }
  }

  return res;
}
