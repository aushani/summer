#include "hilbert_map.h"

#include <math.h>
#include <chrono>
#include <random>
#include <map>
#include <cstring>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <curand.h>
#include <curand_kernel.h>

struct MetaPoint {

  Point p;
  float label;
  int valid;

  int bucket_idx;
  int subbucket_idx;
};

struct DeviceData {

  float *w;

  // Params
  float learning_rate = 0.1;
  float min = -25.0;
  float max =  25.0;

  int inducing_points_n_dim = 100;
  float kernel_width_meters = 1.0;
  float kernel_width_meters_sq = kernel_width_meters * kernel_width_meters;

  // Buckets (for processing in parallel without data collision)
  int bucket_n_dim = 100;
  int subbucket_n_dim = 5;

  // Observations
  float max_range = 100.0;
  float meters_per_observation = 1.0;

  // Computed params
  int kernel_width_xm;
  float bucket_size;
  float inducing_point_step;

  // Kernel table
  float *kernel_table;
  const float kernel_table_scale = 1024; // ~ 1mm resolution
  float kernel_table_resolution = 1.0/kernel_table_scale;
  int kernel_table_n_dim;

  // Raw data we make observations from
  Point *hits = NULL;
  Point *origins = NULL;
  int n_obs;

  // Data points we're considering
  MetaPoint *mp = NULL;
  int n_data;

  // For indexing into buckets
  int *bucket_starts = NULL;
  int n_buckets;
  int n_subbuckets;

  // TODO
  float *d_res = NULL;

  bool own_memory;

  DeviceData(const std::vector<Point> &raw_hits, const std::vector<Point> &raw_origins) {
    auto tic_setup = std::chrono::steady_clock::now();

    // Params
    inducing_point_step = (max - min)/inducing_points_n_dim;
    kernel_width_xm = (kernel_width_meters / inducing_point_step + 1)*2 + 1;
    printf("Kernel width: %d pixels\n", kernel_width_xm);

    // Each subblock must be twice the support size (in meters) of the kernel away from any other subblock
    // that runs concurrently.
    bucket_size = 2.0 * kernel_width_meters * subbucket_n_dim / (subbucket_n_dim - 1);

    // Init w
    int n_inducing_points = inducing_points_n_dim * inducing_points_n_dim;
    cudaMalloc(&w, sizeof(float)*n_inducing_points);
    cudaMemset(w, 0, sizeof(float)*n_inducing_points);

    // Alloc kernel table
    kernel_table_n_dim = ceil(kernel_width_meters / kernel_table_resolution);
    cudaMalloc(&kernel_table, sizeof(float)*kernel_table_n_dim*kernel_table_n_dim);

    // Copy data to device
    cudaMalloc(&hits, sizeof(Point)*raw_hits.size());
    cudaMalloc(&origins, sizeof(Point)*raw_origins.size());

    cudaMemcpy(hits, raw_hits.data(), sizeof(Point)*raw_hits.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(origins, raw_origins.data(), sizeof(Point)*raw_origins.size(), cudaMemcpyHostToDevice);

    // malloc workspace
    int max_obs_per_point = ceil(max_range / meters_per_observation) + 1;
    n_obs = raw_hits.size();
    n_data = n_obs * max_obs_per_point;
    cudaMalloc(&mp, sizeof(MetaPoint)*n_data);

    n_buckets = bucket_n_dim * bucket_n_dim;
    n_subbuckets = subbucket_n_dim * subbucket_n_dim;
    cudaMalloc(&bucket_starts, sizeof(int)*n_buckets*n_subbuckets);

    // TODO
    cudaMalloc(&d_res, 1*sizeof(float));

    own_memory = true;

    auto toc_setup = std::chrono::steady_clock::now();
    auto t_setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_setup - tic_setup);
    printf("\tTook %ld ms to setup device data\n", t_setup_ms.count());
  }

  void cleanup() {
    if (kernel_table) cudaFree(kernel_table);
    if (hits) cudaFree(hits);
    if (origins) cudaFree(origins);
    if (mp) cudaFree(mp);
    if (bucket_starts) cudaFree(bucket_starts);
    if (d_res) cudaFree(d_res);
  }

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

struct IsInvalidMP {
  __host__ __device__ bool operator()(const MetaPoint &mp) const {
    return mp.valid == 0;
  }
};

// Forward declaration of kernels
__global__ void precompute_kernel_table(DeviceData data);
__global__ void make_observations(DeviceData data, unsigned int seed);
__global__ void perform_w_update_buckets(DeviceData data, int subbucket);
__global__ void populate_meta_info(DeviceData data);
__global__ void compute_bucket_indicies(DeviceData data);
__global__ void compute_subbucket_indicies(DeviceData data);

HilbertMap::HilbertMap(const std::vector<Point> &hits, const std::vector<Point> &origins) :
  data_(new DeviceData(hits, origins)) {

  // Precompute kernel table
  auto tic_kernel = std::chrono::steady_clock::now();
  dim3 threads_dim;
  threads_dim.x = 32;
  threads_dim.y = 32;
  threads_dim.z = 1;
  dim3 blocks_dim;
  blocks_dim.x = ceil(data_->kernel_table_n_dim/threads_dim.x);
  blocks_dim.y = ceil(data_->kernel_table_n_dim/threads_dim.y);
  blocks_dim.z = 1;
  precompute_kernel_table<<<threads_dim, blocks_dim>>>(*data_);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_kernel = std::chrono::steady_clock::now();
  auto t_kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(toc_kernel - tic_kernel);
  printf("\tPrecomputed %dx%d kernel lookup table in %5.3f ms with %dx%d threads and %dx%d blocks\n",
      data_->kernel_table_n_dim, data_->kernel_table_n_dim, t_kernel_us.count()/1000.0,
      threads_dim.x, threads_dim.y, blocks_dim.x, blocks_dim.y);

  // Make observations from raw hits
  auto tic_obs = std::chrono::steady_clock::now();
  int threads = 512;
  int blocks = data_->n_obs / threads + 1;
  make_observations<<<threads, blocks>>>(*data_, 0);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_obs = std::chrono::steady_clock::now();
  auto t_obs_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_obs - tic_obs);
  printf("\tMade observations in %ld ms\n", t_obs_ms.count());

  // Remove invalid observations
  auto tic_remove = std::chrono::steady_clock::now();
  thrust::device_ptr<MetaPoint> dp_mp(data_->mp);
  auto dp_mp_end = thrust::remove_if(dp_mp, dp_mp + data_->n_data, IsInvalidMP());
  data_->n_data = dp_mp_end - dp_mp;
  auto toc_remove = std::chrono::steady_clock::now();
  auto t_remove_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_remove - tic_remove);
  printf("\tRemoved invalid in %ld ms\n", t_remove_ms.count());

  // Compute meta information for points
  threads = 512;
  blocks = data_->n_data / threads + 1;
  populate_meta_info<<<blocks, threads>>>(*data_);

  // Sort points by bucket to enable processing in parallel without memory collisions on w
  auto tic_sort = std::chrono::steady_clock::now();
  thrust::device_ptr<MetaPoint> dp_data(data_->mp);
  thrust::sort(dp_data, dp_data + data_->n_data, PointBucketComparator());
  auto toc_sort = std::chrono::steady_clock::now();
  auto t_sort_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_sort - tic_sort);
  printf("\tSorted points into buckets in %ld ms\n", t_sort_ms.count());

  // Find bucket start/end indexes
  threads = 512;
  blocks = data_->n_buckets / threads + 1;
  compute_bucket_indicies<<<blocks, threads>>>(*data_);
  compute_subbucket_indicies<<<blocks, threads>>>(*data_);

  auto tic_learn = std::chrono::steady_clock::now();
  int epochs = 1;
  for (int i=0; i<epochs; i++) {
    for (int subbucket=0; subbucket < data_->n_subbuckets; subbucket++) {
      //auto tic_w = std::chrono::steady_clock::now();
      dim3 threads;
      threads.x = data_->kernel_width_xm;
      threads.y = data_->kernel_width_xm;
      threads.z = 1;
      int blocks = data_->n_buckets;
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
  auto toc_learn = std::chrono::steady_clock::now();
  auto t_learn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_learn - tic_learn);
  printf("\tLearned w in %ld ms\n", t_learn_ms.count());
}

HilbertMap::~HilbertMap() {
  data_->cleanup();
}

__device__ float k_sparse_compute(DeviceData &data, float dx, float dy) {
  float d2 = dx*dx + dy*dy;

  if (d2 > data.kernel_width_meters_sq)
    return 0;

  float r = sqrt(d2);

  // Apply kernel width
  r /= data.kernel_width_meters;

  float t = 2 * M_PI * r;

  return (2 + cosf(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sinf(t);
}

__device__ float k_sparse_lookup(DeviceData &data, float dx, float dy) {
  int idx = llrintf(abs(dx)*data.kernel_table_scale);
  int idy = llrintf(abs(dy)*data.kernel_table_scale);

  if (idx >= data.kernel_table_n_dim || idy >= data.kernel_table_n_dim)
    return 0;

  return data.kernel_table[idx*data.kernel_table_n_dim + idy];
}

__global__ void precompute_kernel_table(DeviceData data) {

  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int idx = bidx*nx + tidx;
  const int idy = bidy*ny + tidy;

  if (idx >= data.kernel_table_n_dim || idy >= data.kernel_table_n_dim)
    return;

  float dx = idx*data.kernel_table_resolution;
  float dy = idy*data.kernel_table_resolution;

  data.kernel_table[idx*data.kernel_table_n_dim + idy] = k_sparse_compute(data, dx, dy);
}

__global__ void make_observations(DeviceData data, unsigned int seed) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  const int idx = bidx*threads + tidx;

  if (idx >= data.n_obs)
    return;

  // Init curand state
  curandState_t curand_state;
  curand_init(seed, bidx, 0, &curand_state);

  int max_obs_per_point = ceil(data.max_range / data.meters_per_observation) + 1;

  Point origin = data.origins[idx];
  Point hit = data.hits[idx];

  // Add hit
  data.mp[idx*max_obs_per_point].p = hit;
  data.mp[idx*max_obs_per_point].label = 1.0;
  data.mp[idx*max_obs_per_point].valid = 1;

  // Draw random free space samples
  float dx = hit.x - origin.x;
  float dy = hit.y - origin.y;

  float range = sqrt(dx*dx + dy*dy);

  for (int i=1; i<max_obs_per_point; i++) {

    // TODO make random
    if (i < range) {
      float rand = 1.0f - curand_uniform(&curand_state); // because the range is (0, 1.0]
      data.mp[idx*max_obs_per_point + i].p.x = origin.x + rand*dx;
      data.mp[idx*max_obs_per_point + i].p.y = origin.y + rand*dy;
      data.mp[idx*max_obs_per_point + i].label = -1.0;
      data.mp[idx*max_obs_per_point + i].valid = 1;
    } else {
      // Dummy observation
      data.mp[idx*max_obs_per_point + i].valid = 0;
    }
  }
}

__global__ void populate_meta_info(DeviceData data) {

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  const int idx = bidx*threads + tidx;

  if (idx >= data.n_data)
    return;

  if (!data.mp[idx].valid) {
    // Flag as invalid bucket
    data.mp[idx].bucket_idx = -1;
    data.mp[idx].subbucket_idx = -1;
    return;
  }

  Point p = data.mp[idx].p;

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

  const float inducing_points_scale = 1.0 / data.inducing_point_step;
  const float kwx2 = data.kernel_width_xm/2.0;

  const int idx_phi = tidx*ny + tidy;
  const int n2 = nx*ny;

  for (int data_idx = start; data_idx < end; ++data_idx) {

    Point x = data.mp[data_idx].p;
    float y = data.mp[data_idx].label;

    int i0 = (x.x - data.min) * inducing_points_scale - kwx2;
    int j0 = (x.y - data.min) * inducing_points_scale - kwx2;

    int i = i0 + tidx;
    int j = j0 + tidy;

    // Evaluate kernel
    int idx_w = i*data.inducing_points_n_dim + j;
    float x_m_x = i*data.inducing_point_step + data.min;
    float x_m_y = j*data.inducing_point_step + data.min;
    float k = k_sparse_lookup(data, x.x - x_m_x, x.y - x_m_y);
    if (i >=0 && i < data.inducing_points_n_dim && j>=0 && j<data.inducing_points_n_dim) {
      phi_sparse[idx_phi] = data.w[idx_w]*k;
    } else {
      phi_sparse[idx_phi] = 0;
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
      int idx_to_add = idx_phi + idx_offset;
      if (idx_to_add < n2)
        phi_sparse[idx_phi] += phi_sparse[idx_to_add];
      __syncthreads();
    }

    float wTphi = phi_sparse[0];

    float c = -y * 1.0/(1.0 + __expf(y * wTphi));
    //float c = -y * wTphi;

    data.w[idx_w] -= data.learning_rate * c * k;
  }
}

__global__ void compute_occupancy(DeviceData data, Point x) {

  // Figure out where this thread is
  const int nx = blockDim.x;
  const int ny = blockDim.y;

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  extern __shared__ float phi_sparse[];

  int i0 = (x.x - data.min) / data.inducing_point_step - data.kernel_width_xm/2.0;
  int j0 = (x.y - data.min) / data.inducing_point_step - data.kernel_width_xm/2.0;

  int i = i0 + tidx;
  int j = j0 + tidy;

  // Evaluate kernel
  float x_m_x = i*data.inducing_point_step + data.min;
  float x_m_y = j*data.inducing_point_step + data.min;
  float k = k_sparse_lookup(data, x.x - x_m_x, x.y - x_m_y);
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

  data.d_res[0] = 1.0 / (1.0 + __expf(wTphi));
}

float HilbertMap::get_occupancy(Point p) {

  dim3 threads;
  threads.x = data_->kernel_width_xm;
  threads.y = data_->kernel_width_xm;
  threads.z = 1;
  size_t blocks = 1;
  int sm_size = threads.x*threads.y*sizeof(float);
  //printf("Running with %dx%d threads and %d blocks and %ld sm_size\n",
  //    threads.x, threads.y, blocks, sm_size);
  compute_occupancy<<<blocks, threads, sm_size>>>(*data_, p);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }

  float res;
  cudaMemcpy(&res, data_->d_res, 1*sizeof(float), cudaMemcpyDeviceToHost);
  return res;
}
