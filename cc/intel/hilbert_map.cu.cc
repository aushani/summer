#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>
#include <map>
#include <cstring>

#include <thrust/device_ptr.h>

struct DeviceData {

  float *w;

  // Params
  float learning_rate;
  int kernel_width;
  float min;
  float max;

  // For inducing point computations
  int inducing_points_n_dim;
  float inducing_point_step;

  // Data points we're considering
  Point *points;
  float *labels;
  int n_data;

  float *d_res;

};

// Forward declaration of kernels
__global__ void perform_w_update(DeviceData data);

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<float> occupancies) {

  auto tic = std::chrono::steady_clock::now();

  data_ = new DeviceData;

  // Init w
  cudaMalloc(&data_->w, sizeof(float)*n_inducing_points);
  cudaMemset(data_->w, 0, sizeof(float)*n_inducing_points);

  // Copy data to device
  cudaMalloc(&data_->points, sizeof(Point)*points.size());
  cudaMalloc(&data_->labels, sizeof(float)*occupancies.size());

  cudaMemcpy(data_->points, points.data(), sizeof(Point)*points.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(data_->labels, occupancies.data(), sizeof(float)*occupancies.size(), cudaMemcpyHostToDevice);

  data_->n_data = points.size();

  // Params
  data_->learning_rate = 0.1;
  data_->kernel_width = 7;
  data_->min = -25.0;
  data_->max =  25.0;
  data_->inducing_points_n_dim = inducing_points_n_dim;
  data_->inducing_point_step = (data_->max - data_->min)/inducing_points_n_dim;

  // TODO
  cudaMalloc(&data_->d_res, 1*sizeof(float));

  int epochs = 1;
  for (int i=0; i<epochs; i++) {
    dim3 threads;
    threads.x = data_->kernel_width;
    threads.y = data_->kernel_width;
    int blocks = 1;
    int sm_size = threads.x*threads.y*sizeof(float);
    perform_w_update<<<blocks, threads, sm_size>>>(*data_);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("Whoops\n");
    }
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

__global__ void perform_w_update(DeviceData data) {

  // Figure out where this thread is
  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  extern __shared__ float phi_sparse[];

  for (int data_idx = 0; data_idx < data.n_data; data_idx++) {

    Point x = data.points[data_idx];
    float y = data.labels[data_idx];

    int i0 = (x.x - data.min) / data.inducing_point_step - data.kernel_width/2.0;
    int j0 = (x.y - data.min) / data.inducing_point_step - data.kernel_width/2.0;

    int i = i0 + tidx;
    int j = j0 + tidy;

    // Evaluate kernel
    int idx_w = i*data.inducing_points_n_dim + j;
    int idx = tidx*ny + tidy;
    float x_m_x = i*data.inducing_point_step + data.min;
    float x_m_y = j*data.inducing_point_step + data.min;
    float k = k_sparse(x, x_m_x, x_m_y);
    if (i >=0 && i < data.inducing_points_n_dim && j>=0 && j<data.inducing_points_n_dim) {
      phi_sparse[idx] = data.w[idx_w]*k;
    } else {
      phi_sparse[idx] = 0;
    }

    __syncthreads();

    // Compute gradient and update w

    /*
    float wTphi = 0.0;
    for (int i=0; i<nx*ny; i++) {
      wTphi += phi_sparse[i];
    }
    */
    // Reduction to sum
    for (int idx_offset = 1; idx_offset<nx*ny; idx_offset<<=1) {
      int idx_to_add = idx + idx_offset;
      if (idx_to_add < nx*ny)
        phi_sparse[idx] += phi_sparse[idx_to_add];
      __syncthreads();
    }

    float wTphi = phi_sparse[0];

    float c = -y * 1.0/(1.0 + expf(y * wTphi));

    data.w[idx_w] -= data.learning_rate * c * k;
  }
}

__global__ void compute_occupancy(DeviceData data) {

  // Figure out where this thread is
  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  extern __shared__ float phi_sparse[];

  Point x = data.points[0];

  int i0 = (x.x - data.min) / data.inducing_point_step - data.kernel_width/2.0;
  int j0 = (x.y - data.min) / data.inducing_point_step - data.kernel_width/2.0;

  int i = i0 + tidx;
  int j = j0 + tidy;

  // Evaluate kernel
  float x_m_x = i*data.inducing_point_step + data.min;
  float x_m_y = j*data.inducing_point_step + data.min;
  float k = k_sparse(data.points[0], x_m_x, x_m_y);
  phi_sparse[tidx*ny + tidy] = k;

  __syncthreads();

  float wTphi = 0.0;
  for (int di=0; di<nx; di++) {
    int w_i = i0 + di;
    if (w_i < 0 || w_i >= data.inducing_points_n_dim)
      continue;
    for (int dj=0; dj<ny; dj++) {
      int w_j = j0 + dj;
      if (w_j < 0 || w_j >= data.inducing_points_n_dim)
        continue;
      int idx_w = w_i*data.inducing_points_n_dim + w_j;
      int idx_sparse = di*ny + dj;
      wTphi += data.w[idx_w] * phi_sparse[idx_sparse];
    }
  }

  int idx_w = i0*data.inducing_points_n_dim + j0;
  data.d_res[0] = 1.0 / (1.0 + expf(wTphi));
}

float HilbertMap::get_occupancy(Point p) {

  // ugly
  cudaMemcpy(data_->points, &p, sizeof(Point), cudaMemcpyHostToDevice);

  dim3 threads;
  threads.x = data_->kernel_width;
  threads.y = data_->kernel_width;
  size_t blocks = 1;
  int sm_size = threads.x*threads.y*sizeof(float);
  //printf("Running with %dx%d threads and %d blocks and %ld sm_size\n",
  //    threads.x, threads.y, blocks, sm_size);
  compute_occupancy<<<blocks, threads, sm_size>>>(*data_);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }

  float res;
  cudaMemcpy(&res, data_->d_res, 1*sizeof(float), cudaMemcpyDeviceToHost);
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
