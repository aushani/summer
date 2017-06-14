#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>
#include <map>
#include <cstring>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

struct MetaPoint {

  Point p;
  float label;

  int bucket_idx;
  int subbucket_idx;
};

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
  MetaPoint *mp;
  int n_data;

  // Buckets (for processing in parallel without data collision)
  int bucket_n_dim;
  int subbucket_n_dim;
  float bucket_size;

  int *bucket_starts;

  float *d_res;

};

struct PointBucketComparator {
  __device__ bool operator()(const MetaPoint &lhs,
                             const MetaPoint &rhs) const {

    if (lhs.bucket_idx != rhs.bucket_idx) {
      return lhs.bucket_idx < rhs.bucket_idx;
    }

    return lhs.subbucket_idx < rhs.subbucket_idx;
  }
};

// Forward declaration of kernels
__global__ void perform_w_update(DeviceData data);
__global__ void perform_w_update_buckets(DeviceData data, int subbucket);
__global__ void populate_meta_info(DeviceData data);
__global__ void compute_bucket_indicies(DeviceData data);
__global__ void compute_subbucket_indicies(DeviceData data);

HilbertMap::HilbertMap(std::vector<Point> points, std::vector<float> occupancies) {

  auto tic_total = std::chrono::steady_clock::now();

  data_ = new DeviceData;

  // Params
  data_->learning_rate = 0.1;
  data_->min = -25.0;
  data_->max =  25.0;
  data_->inducing_points_n_dim = 100;
  data_->inducing_point_step = (data_->max - data_->min)/data_->inducing_points_n_dim;
  data_->kernel_width = (1.0 / data_->inducing_point_step + 1)*2 + 1;
  printf("Kernel width %d with a step size of %5.3f\n", data_->kernel_width, data_->inducing_point_step);

  data_->bucket_n_dim = 100;
  data_->subbucket_n_dim = 5;

  // Each subblock must be twice the support size (in meters) of the kernel away from any other subblock
  // that runs concurrently.
  double kernel_width_meters = 1.0;
  data_->bucket_size = 2.0 * kernel_width_meters * data_->subbucket_n_dim / (data_->subbucket_n_dim - 1);

  // Init w
  auto tic_w = std::chrono::steady_clock::now();
  int n_inducing_points = data_->inducing_points_n_dim * data_->inducing_points_n_dim;
  cudaMalloc(&data_->w, sizeof(float)*n_inducing_points);
  cudaMemset(data_->w, 0, sizeof(float)*n_inducing_points);
  auto toc_w = std::chrono::steady_clock::now();
  auto t_w_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_w - tic_w);
  printf("\tTook %ld ms to init w\n", t_w_ms.count());

  // Copy data to device
  auto tic_data = std::chrono::steady_clock::now();
  cudaMalloc(&data_->points, sizeof(Point)*points.size());
  cudaMalloc(&data_->labels, sizeof(float)*occupancies.size());

  cudaMemcpy(data_->points, points.data(), sizeof(Point)*points.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(data_->labels, occupancies.data(), sizeof(float)*occupancies.size(), cudaMemcpyHostToDevice);

  cudaMalloc(&data_->mp, sizeof(MetaPoint)*points.size());

  data_->n_data = points.size();
  auto toc_data = std::chrono::steady_clock::now();
  auto t_data_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_data - tic_data);
  printf("\tTook %ld ms to init data\n", t_data_ms.count());

  // TODO
  cudaMalloc(&data_->d_res, 1*sizeof(float));

  int n_buckets = data_->bucket_n_dim * data_->bucket_n_dim;
  int n_subbuckets = data_->subbucket_n_dim * data_->subbucket_n_dim;
  cudaMalloc(&data_->bucket_starts, sizeof(int)*n_buckets*n_subbuckets);

  // Compute meta information for points
  auto tic_meta = std::chrono::steady_clock::now();
  int threads = 512;
  int blocks = data_->n_data / threads + 1;
  populate_meta_info<<<blocks, threads>>>(*data_);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_meta = std::chrono::steady_clock::now();
  auto t_meta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_meta - tic_meta);
  printf("Computed point buckets in %ld ms\n", t_meta_ms.count());

  // Sort points by bucket to enable processing in parallel without memory collisions on w
  auto tic_sort = std::chrono::steady_clock::now();
  thrust::device_ptr<MetaPoint> dp_data(data_->mp);
  thrust::sort(dp_data, dp_data + data_->n_data, PointBucketComparator());
  auto toc_sort = std::chrono::steady_clock::now();
  auto t_sort_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_sort - tic_sort);
  printf("Sorted points into buckets in %ld ms\n", t_sort_ms.count());

  // Find bucket start/end indexes
  auto tic_starts = std::chrono::steady_clock::now();
  threads = 512;
  blocks = n_buckets / threads + 1;
  compute_bucket_indicies<<<blocks, threads>>>(*data_);
  compute_subbucket_indicies<<<blocks, threads>>>(*data_);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_starts = std::chrono::steady_clock::now();
  auto t_starts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_starts - tic_starts);
  printf("Computed point bucket starts in %ld ms\n", t_meta_ms.count());

  int epochs = 1;
  for (int i=0; i<epochs; i++) {
    for (int subbucket=0; subbucket < n_subbuckets; subbucket++) {
      //auto tic_w = std::chrono::steady_clock::now();
      dim3 threads;
      threads.x = data_->kernel_width;
      threads.y = data_->kernel_width;
      int blocks = n_buckets;
      int sm_size = threads.x*threads.y*sizeof(float);
      //perform_w_update<<<blocks, threads, sm_size>>>(*data_);
      perform_w_update_buckets<<<blocks, threads, sm_size>>>(*data_, subbucket);
      //cudaError_t err = cudaDeviceSynchronize();
      //if (err != cudaSuccess) {
      //  printf("Whoops\n");
      //}
      //auto toc_w = std::chrono::steady_clock::now();
      //auto t_w_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_w - tic_w);
      //printf("\tTook %02ld ms to update w on subbucket %d with %d blocks and %dx%d threads\n",
      //    t_w_ms.count(), subbucket, blocks, threads.x, threads.y);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_total = std::chrono::steady_clock::now();
  auto t_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_total - tic_total);
  printf("Done learning w in %ld ms\n", t_total_ms.count());

}

HilbertMap::~HilbertMap() {
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

__global__ void populate_meta_info(DeviceData data) {

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  const int idx = bidx*threads + tidx;

  if (idx >= data.n_data)
    return;

  Point p = data.points[idx];
  float label = data.labels[idx];

  data.mp[idx].p = p;
  data.mp[idx].label = label;

  // Compute bucket
  int bucket_i = llrintf(p.x/data.bucket_size);
  int bucket_j = llrintf(p.y/data.bucket_size);

  bucket_i %= data.bucket_n_dim;
  bucket_j %= data.bucket_n_dim;

  if (bucket_i < 0)
    bucket_i += data.bucket_n_dim;

  if (bucket_j < 0)
    bucket_j += data.bucket_n_dim;

  data.mp[idx].bucket_idx = bucket_j * data.bucket_n_dim + bucket_j;

  // Compute subbucket
  float x0 = bucket_i * data.bucket_size;
  float y0 = bucket_j * data.bucket_size;
  float subbucket_size = data.bucket_size/data.subbucket_n_dim;
  int subbucket_i = llrintf((p.x - x0)/subbucket_size);
  int subbucket_j = llrintf((p.y - y0)/subbucket_size);

  subbucket_i %= data.subbucket_n_dim;
  subbucket_j %= data.subbucket_n_dim;

  if (subbucket_i < 0)
    subbucket_i += data.subbucket_n_dim;

  if (subbucket_j < 0)
    subbucket_j += data.subbucket_n_dim;

  data.mp[idx].subbucket_idx = subbucket_i * data.subbucket_n_dim + subbucket_j;
}

__global__ void compute_bucket_indicies(DeviceData data) {

  // Figure out where this thread is
  const int threads = blockDim.x;

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  int bucket_idx = bidx * threads + tidx;

  int n_subbuckets = data.subbucket_n_dim * data.subbucket_n_dim;

  if (bucket_idx >= data.bucket_n_dim * data.bucket_n_dim)
    return;

  // First find starting index of "bucket"
  // binary search
  int a = 0;
  int b = data.n_data;

  while (a<b-1) {

    int mid = (a + b) >> 1;

    int idx = data.mp[mid].bucket_idx;

    if (idx < bucket_idx) {
      a = mid;
    } else {
      b = mid;
    }
  }

  // Check to make sure we're not starting at 0
  if (data.mp[a].bucket_idx == bucket_idx)
    data.bucket_starts[bucket_idx * n_subbuckets] = a;
  else
    data.bucket_starts[bucket_idx * n_subbuckets] = b;

  // Verify
  //int start = data.bucket_starts[bucket_idx];
  //if (data.mp[start].bucket_idx >= bucket_idx && (start==0 || data.mp[start-1].bucket_idx < bucket_idx)) {
  //  printf("Good\n");
  //} else {
  //  printf("Bucket %d starts at index %d - %d\n", bucket_idx, a, b);
  //  printf(" Bucket %02d [a] = %d (a=%d)\n", bucket_idx, data.mp[a].bucket_idx, a);
  //  __syncthreads();
  //  printf(" Bucket %02d [b] = %d (b=%d)\n", bucket_idx, data.mp[b].bucket_idx, b);
  //}
}

__global__ void compute_subbucket_indicies(DeviceData data) {

  // Figure out where this thread is
  const int threads = blockDim.x;

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  int bucket_idx = bidx * threads + tidx;

  int n_subbuckets = data.subbucket_n_dim * data.subbucket_n_dim;

  if (bucket_idx >= data.bucket_n_dim * data.bucket_n_dim)
    return;

  // First find starting index of subbucket
  // binary search
  for (int subbucket = 1; subbucket < n_subbuckets; subbucket++) {
    int a = data.bucket_starts[bucket_idx * n_subbuckets];
    int b = data.n_data;

    if (bucket_idx + 1 < data.bucket_n_dim*data.bucket_n_dim) {
      b = data.bucket_starts[(bucket_idx + 1) * n_subbuckets];
    }

    while (a<b-1) {

      int mid = (a + b) >> 1;

      int idx = data.mp[mid].subbucket_idx;

      if (idx < subbucket) {
        a = mid;
      } else {
        b = mid;
      }
    }

    // Check to make sure we're not starting at 0
    if (data.mp[a].subbucket_idx == subbucket)
      data.bucket_starts[bucket_idx * n_subbuckets + subbucket] = a;
    else
      data.bucket_starts[bucket_idx * n_subbuckets + subbucket] = b;
  }

  // Verify
  //int start = data.bucket_starts[bucket_idx];
  //if (data.mp[start].bucket_idx >= bucket_idx && (start==0 || data.mp[start-1].bucket_idx < bucket_idx)) {
  //  printf("Good\n");
  //} else {
  //  printf("Bucket %d starts at index %d - %d\n", bucket_idx, a, b);
  //  printf(" Bucket %02d [a] = %d (a=%d)\n", bucket_idx, data.mp[a].bucket_idx, a);
  //  __syncthreads();
  //  printf(" Bucket %02d [b] = %d (b=%d)\n", bucket_idx, data.mp[b].bucket_idx, b);
  //}
}

__global__ void perform_w_update_buckets(DeviceData data, int subbucket) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;

  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  extern __shared__ float phi_sparse[];

  int n_buckets = data.bucket_n_dim * data.bucket_n_dim;
  int n_subbuckets = data.subbucket_n_dim * data.subbucket_n_dim;

  if (bidx >= n_buckets)
    return;

  const int start = data.bucket_starts[bidx*n_subbuckets + subbucket];
  int end = data.n_data;

  if (bidx + 1 < n_buckets || subbucket + 1 < n_subbuckets) {
    end = data.bucket_starts[bidx*n_subbuckets + subbucket + 1];
  }

  for (int data_idx = start; data_idx < end; data_idx++) {

    Point x = data.mp[data_idx].p;
    float y = data.mp[data_idx].label;

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
