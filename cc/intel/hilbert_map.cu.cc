#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>
#include <map>
#include <cstring>

#include <thrust/device_ptr.h>

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<float> occupancies) {

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
    float y = occupancies[idx];
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

__device__ float k_sparse(Point p1, Point p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;

  float d2 = dx*dx + dy*dy;

  if (d2 > 1.0)
    return 0;

  float r = sqrt(d2);

  float t = 2 * M_PI * r;

  return (2 + cos(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sin(t);
}

struct GPUData {

  float *w;

  float learning_rate;

  // For inducing point computations
  int i0;
  int j0;
  Point x_m_0;
  float inducing_point_step;

  // Data point we're evaluating
  Point p;
  float y;
};

__global__ void perform_w_update(GPUData data) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int threads = blockDim.x;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i = data.i0 + tidx;
  const int j = data.j0 + tidy;
}

void HilbertMap::update_w(Point x, float y) {
  float learning_rate = 0.1;

  // Where does this point actually have support?
  std::vector<int> support_idx = get_inducing_points_with_support(x);

  // Evaluate kernel
  std::map<int, float> phi_sparse;
  std::vector<int> nz;
  for (int i : support_idx) {
    float k = k_sparse(x, get_inducing_point(i));
    if (k>1e-5) {
      phi_sparse[i] = k;
      nz.push_back(i);
    }
  }

  // Compute gradient and update w
  float wTphi = 0.0;
  for (int i : nz) {
    wTphi += w_[i] * phi_sparse[i];
  }
  float c = -y * 1.0/(1.0 + expf(y * wTphi));

  // TODO no regularization

  for (int i : nz) {
    w_[i] -= learning_rate * c * phi_sparse[i];
  }
}

float HilbertMap::get_occupancy(Point p) {
  // Where does this point actually have support?
  std::vector<int> support_idx = get_inducing_points_with_support(p);

  // Evaluate
  float wTphi = 0.0;
  for (int i : support_idx) {
    wTphi += w_[i] * k_sparse(p, get_inducing_point(i));
  }
  return 1.0 / (1.0 + expf(wTphi));
}

Point HilbertMap::get_inducing_point(int idx) {
  float min = -25.0;
  float max =  25.0;

  int i = idx / inducing_points_n_dim;
  int j = idx % inducing_points_n_dim;

  float x = min + i * (max - min) / inducing_points_n_dim;
  float y = min + j * (max - min) / inducing_points_n_dim;

  return Point(x, y);
}

std::vector<int> HilbertMap::get_inducing_points_with_support(Point p) {
  float min = -25.0;
  float max =  25.0;

  float step = (max - min)/inducing_points_n_dim;

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
