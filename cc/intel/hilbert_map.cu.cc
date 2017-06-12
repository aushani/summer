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
  cudaMalloc(&d_w_, sizeof(float)*n_inducing_points);
  cudaMemset(d_w_, 0, sizeof(float)*n_inducing_points);

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
  cudaMalloc(&d_w_, sizeof(float)*n_inducing_points);
  cudaMemcpy(d_w_, hm.d_w_, sizeof(float)*n_inducing_points, cudaMemcpyDeviceToDevice);
}

HilbertMap::~HilbertMap() {
  if (d_w_)
    cudaFree(d_w_);
}

__device__ float k_sparse(Point p1, float x_m_x, float x_m_y) {
  float dx = p1.x - x_m_x;
  float dy = p1.y - x_m_y;

  float d2 = dx*dx + dy*dy;

  if (d2 > 1.0)
    return 0;

  float r = sqrt(d2);

  float t = 2 * M_PI * r;

  return (2 + cosf(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sinf(t);
}

struct GPUData {

  float *w;

  float learning_rate;

  // For inducing point computations
  int i0;
  int j0;
  int inducing_points_n_dim;
  float inducing_point_step;

  // Data point we're evaluating
  Point p;
  float y;

  float *d_res;

  GPUData() {;}
};

__global__ void perform_w_update(GPUData data) {

  // Figure out where this thread is
  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i = data.i0 + tidx;
  const int j = data.j0 + tidy;

  extern __shared__ float phi_sparse[];

  // Evaluate kernel
  float x_m_x = i*data.inducing_point_step - 25.0;
  float x_m_y = j*data.inducing_point_step - 25.0;
  float k = k_sparse(data.p, x_m_x, x_m_y);
  phi_sparse[tidx*ny + tidy] = k;

  __syncthreads();

  // Compute gradient and update w
  float wTphi = 0.0;
  for (int di=0; di<nx; di++) {
    int w_i = data.i0 + di;
    if (w_i < 0 || w_i >= data.inducing_points_n_dim)
      continue;
    for (int dj=0; dj<ny; dj++) {
      int w_j = data.j0 + dj;
      if (w_j < 0 || w_j >= data.inducing_points_n_dim)
        continue;
      int idx_w = w_i*data.inducing_points_n_dim + w_j;
      int idx_sparse = di*ny + dj;
      wTphi += data.w[idx_w] * phi_sparse[idx_sparse];
    }
  }
  float c = -data.y * 1.0/(1.0 + expf(data.y * wTphi));

  // TODO no regularization

  int idx_w = i*data.inducing_points_n_dim + j;
  data.w[idx_w] -= data.learning_rate * c * k;
}

__global__ void compute_occupancy(GPUData data) {

  // Figure out where this thread is
  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int i = data.i0 + tidx;
  const int j = data.j0 + tidy;

  extern __shared__ float phi_sparse[];

  // Evaluate kernel
  float x_m_x = i*data.inducing_point_step - 25.0;
  float x_m_y = j*data.inducing_point_step - 25.0;
  float k = k_sparse(data.p, x_m_x, x_m_y);
  phi_sparse[tidx*ny + tidy] = k;

  __syncthreads();

  float wTphi = 0.0;
  for (int di=0; di<nx; di++) {
    int w_i = data.i0 + di;
    if (w_i < 0 || w_i >= data.inducing_points_n_dim)
      continue;
    for (int dj=0; dj<ny; dj++) {
      int w_j = data.j0 + dj;
      if (w_j < 0 || w_j >= data.inducing_points_n_dim)
        continue;
      int idx_w = w_i*data.inducing_points_n_dim + w_j;
      int idx_sparse = di*ny + dj;
      wTphi += data.w[idx_w] * phi_sparse[idx_sparse];
    }
  }

  *data.d_res = 1.0 / (1.0 + expf(wTphi));
}

void HilbertMap::update_w(Point x, float y) {
  float learning_rate = 0.1;

  int kernel_width = 7;

  GPUData gd;
  gd.w = d_w_;
  gd.learning_rate = learning_rate;

  float min = -25.0;
  float max =  25.0;

  gd.inducing_points_n_dim = inducing_points_n_dim;
  gd.inducing_point_step = (max - min)/inducing_points_n_dim;

  gd.i0 = (x.x - min) / gd.inducing_point_step - kernel_width/2;
  gd.j0 = (x.y - min) / gd.inducing_point_step - kernel_width/2;

  gd.p = x;
  gd.y = y;

  dim3 threads;
  threads.x = kernel_width;
  threads.y = kernel_width;
  int blocks = 1;
  int sm_size = threads.x*threads.y*sizeof(float);
  perform_w_update<<<blocks, threads, sm_size>>>(gd);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
}

float HilbertMap::get_occupancy(Point p) {
  int kernel_width = 7;

  GPUData gd;
  gd.w = d_w_;

  cudaMalloc(&gd.d_res, 1*sizeof(float));

  float min = -25.0;
  float max =  25.0;

  gd.inducing_points_n_dim = inducing_points_n_dim;
  gd.inducing_point_step = (max - min)/inducing_points_n_dim;

  gd.i0 = (p.x - min) / gd.inducing_point_step - kernel_width/2;
  gd.j0 = (p.y - min) / gd.inducing_point_step - kernel_width/2;

  gd.p = p;

  dim3 threads;
  threads.x = kernel_width;
  threads.y = kernel_width;
  size_t blocks = 1;
  int sm_size = threads.x*threads.y*sizeof(float);
  //printf("Running with %dx%d threads and %d blocks and %ld sm_size\n",
  //    threads.x, threads.y, blocks, sm_size);
  compute_occupancy<<<blocks, threads, sm_size>>>(gd);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }

  float res;
  cudaMemcpy(&res, gd.d_res, 1*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(gd.d_res);
  return res;
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
