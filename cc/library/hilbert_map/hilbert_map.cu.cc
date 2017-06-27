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

namespace library {
namespace hilbert_map {

// Forward declaration of kernels
__global__ void copy_observations(DeviceData data, Point *points, float *labels);
__global__ void make_observations(DeviceData data, Point *hits, Point *origins, int n_obs, unsigned int seed);
__global__ void perform_w_update_buckets(DeviceData data, int subbucket);
__global__ void populate_meta_info(DeviceData data);
__global__ void compute_bucket_indicies(DeviceData data);
__global__ void compute_subbucket_indicies(DeviceData data);

struct MetaPoint {

  Point p;
  float label;
  int valid;

  int bucket_idx;
  int subbucket_idx;
};

struct IsInvalidMP {
  __host__ __device__ bool operator()(const MetaPoint &mp) const {
    return mp.valid == 0;
  }
};

struct DeviceData {

  float *w;

  Opt opt;

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
  DeviceKernelTable kernel_table;

  // Data points we're considering
  MetaPoint *mp = NULL;
  int n_data;

  // For indexing into buckets
  int *bucket_starts = NULL;
  int n_buckets;
  int n_subbuckets;

  DeviceData(const IKernel &kernel, Opt opt) :
    opt(opt),
    kernel_table(kernel.MakeDeviceKernelTable()) {
    // Params
    inducing_point_step = (opt.max - opt.min)/opt.inducing_points_n_dim;
    kernel_width_xm = (kernel.MaxSupport() / inducing_point_step + 1)*2 + 1;

    // Each subblock must be twice the support size (in meters) of the kernel away from any other subblock
    // that runs concurrently.
    bucket_size = 2.0 * kernel.MaxSupport() * subbucket_n_dim / (subbucket_n_dim - 1);

    // Init w
    int n_inducing_points = opt.inducing_points_n_dim * opt.inducing_points_n_dim;
    cudaMalloc(&w, sizeof(float)*n_inducing_points);
    cudaMemset(w, 0, sizeof(float)*n_inducing_points);
  }

  DeviceData(const std::vector<Point> &raw_hits, const std::vector<Point> &raw_origins, const IKernel &kernel, Opt opt) :
    DeviceData(kernel, opt) {
    // malloc workspace
    int max_obs_per_point = ceil(max_range / meters_per_observation) + 1;
    n_data = raw_hits.size() * max_obs_per_point;
    cudaMalloc(&mp, sizeof(MetaPoint)*n_data);

    // Copy data to device
    Point *d_hits, *d_origins;
    cudaMalloc(&d_hits, sizeof(Point)*raw_hits.size());
    cudaMalloc(&d_origins, sizeof(Point)*raw_origins.size());

    cudaMemcpy(d_hits, raw_hits.data(), sizeof(Point)*raw_hits.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_origins, raw_origins.data(), sizeof(Point)*raw_origins.size(), cudaMemcpyHostToDevice);

    // Make observations from raw hits
    int threads = 512;
    int blocks = raw_hits.size() / threads + 1;
    make_observations<<<threads, blocks>>>(*this, d_hits, d_origins, raw_hits.size(), 0);
    thrust::device_ptr<MetaPoint> dp_mp(mp);
    auto dp_mp_end = thrust::remove_if(dp_mp, dp_mp + n_data, IsInvalidMP());
    n_data = dp_mp_end - dp_mp;

    n_buckets = bucket_n_dim * bucket_n_dim;
    n_subbuckets = subbucket_n_dim * subbucket_n_dim;
    cudaMalloc(&bucket_starts, sizeof(int)*n_buckets*n_subbuckets);

    // Cleanup
    cudaFree(d_hits);
    cudaFree(d_origins);
  }

  DeviceData(const std::vector<Point> &points, const std::vector<float> &labels, const IKernel &kernel, Opt opt) :
    DeviceData(kernel, opt) {
    // malloc workspace
    n_data = points.size();
    cudaMalloc(&mp, sizeof(MetaPoint)*n_data);

    // Copy data to device
    Point *d_points;
    float *d_labels;
    cudaMalloc(&d_points, sizeof(Point)*points.size());
    cudaMalloc(&d_labels, sizeof(float)*labels.size());

    cudaMemcpy(d_points, points.data(), sizeof(Point)*points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), sizeof(float)*labels.size(), cudaMemcpyHostToDevice);

    // Make observations from raw hits
    int threads = 512;
    int blocks = ceil(((double)n_data) / threads);
    copy_observations<<<threads, blocks>>>(*this, d_points, d_labels);

    n_buckets = bucket_n_dim * bucket_n_dim;
    n_subbuckets = subbucket_n_dim * subbucket_n_dim;
    cudaMalloc(&bucket_starts, sizeof(int)*n_buckets*n_subbuckets);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_labels);
  }

  void Cleanup() {
    if (mp) cudaFree(mp);
    if (bucket_starts) cudaFree(bucket_starts);
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

HilbertMap::HilbertMap(DeviceData *data) :
  data_(data) {
  // Compute meta information for points
  int threads = 512;
  int blocks = data_->n_data / threads + 1;
  populate_meta_info<<<blocks, threads>>>(*data_);

  // Sort points by bucket to enable processing in parallel without memory collisions on w
  auto tic_sort = std::chrono::steady_clock::now();
  thrust::device_ptr<MetaPoint> dp_data(data_->mp);
  thrust::sort(dp_data, dp_data + data_->n_data, PointBucketComparator());
  auto toc_sort = std::chrono::steady_clock::now();
  auto t_sort_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_sort - tic_sort);
  //printf("\tSorted points into buckets in %ld ms\n", t_sort_ms.count());

  // Find bucket start/end indexes
  threads = 512;
  blocks = data_->n_buckets / threads + 1;
  compute_bucket_indicies<<<blocks, threads>>>(*data_);
  compute_subbucket_indicies<<<blocks, threads>>>(*data_);

  auto tic_learn = std::chrono::steady_clock::now();
  int epochs = 1;
  for (int i=0; i<epochs; i++) {
    for (int subbucket=0; subbucket < data_->n_subbuckets; subbucket++) {
      auto tic_w = std::chrono::steady_clock::now();
      int threads = 32;
      int blocks = data_->n_buckets;
      int sm_size = threads*sizeof(float);
      perform_w_update_buckets<<<blocks, threads, sm_size>>>(*data_, subbucket);
      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        printf("Whoops\n");
      }
      auto toc_w = std::chrono::steady_clock::now();
      auto t_w_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_w - tic_w);
      printf("\tTook %02ld ms to update w on subbucket %d/%d with %d blocks and %d threads\n",
          t_w_ms.count(), subbucket, data_->n_subbuckets, blocks, threads);
    }
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }
  auto toc_learn = std::chrono::steady_clock::now();
  auto t_learn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_learn - tic_learn);
  //printf("\tLearned w in %ld ms\n", t_learn_ms.count());
}

HilbertMap::HilbertMap(const std::vector<Point> &hits, const std::vector<Point> &origins, const IKernel &kernel, Opt opt)
  : HilbertMap(new DeviceData(hits, origins, kernel, opt)) {
}

HilbertMap::HilbertMap(const std::vector<Point> &points, const std::vector<float> &labels, const IKernel &kernel, Opt opt)
  : HilbertMap(new DeviceData(points, labels, kernel, opt)) {
}

HilbertMap::~HilbertMap() {
  data_->Cleanup();
  data_->kernel_table.Cleanup();
}

__device__ float kernel_lookup(const DeviceData &data, float dx, float dy) {
  float x_kt = dx - data.kernel_table.x0;
  float y_kt = dy - data.kernel_table.y0;

  if (x_kt < 0 || y_kt < 0)
    return 0.0f;

  int idx = llrintf(x_kt*data.kernel_table.scale);
  int idy = llrintf(y_kt*data.kernel_table.scale);

  if (idx >= data.kernel_table.n_dim || idy >= data.kernel_table.n_dim)
    return 0.0f;

  return data.kernel_table.kernel_table[idx*data.kernel_table.n_dim + idy];
}

__global__ void copy_observations(DeviceData data, Point *points, float *labels) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  const int idx = bidx*threads + tidx;

  if (idx >= data.n_data)
    return;

  Point point = points[idx];
  float label = labels[idx];

  // Add hit
  data.mp[idx].p = point;
  data.mp[idx].label = label;
  data.mp[idx].valid = 1;
}

__global__ void make_observations(DeviceData data, Point *hits, Point *origins, int n_obs, unsigned int seed) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  const int idx = bidx*threads + tidx;

  if (idx >= n_obs)
    return;

  // Init curand state
  curandState_t curand_state;
  curand_init(seed, bidx, 0, &curand_state);

  int max_obs_per_point = ceil(data.max_range / data.meters_per_observation) + 1;

  Point hit = hits[idx];
  Point origin = origins[idx];

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

__device__ float compute_wTphi(const DeviceData &data, Point x, int tidx, int threads, float *sh_mem) {

  const int kwxm = data.kernel_width_xm;
  const int num_evals_total = kwxm*kwxm;

  const float inducing_points_scale = 1.0 / data.inducing_point_step;
  const float kwx2 = data.kernel_width_xm/2.0;

  const int i0 = (x.x - data.opt.min) * inducing_points_scale - kwx2;
  const int j0 = (x.y - data.opt.min) * inducing_points_scale - kwx2;

  float thread_phi_sparse = 0.0f;
  for (int eval_num=tidx; eval_num < num_evals_total; eval_num += threads) {

    int i = i0 + (eval_num / kwxm);
    int j = j0 + (eval_num % kwxm);

    int idx_w = i*data.opt.inducing_points_n_dim + j;
    float x_m_x = i*data.inducing_point_step + data.opt.min;
    float x_m_y = j*data.inducing_point_step + data.opt.min;
    float k = kernel_lookup(data, x.x - x_m_x, x.y - x_m_y);
    if (i >=0 && i < data.opt.inducing_points_n_dim && j>=0 && j<data.opt.inducing_points_n_dim) {
      thread_phi_sparse += data.w[idx_w]*k;
    }
  }

  sh_mem[tidx] = thread_phi_sparse;

  __syncthreads();

  // Reduction to sum
  for (int idx_offset = 1; idx_offset<threads; idx_offset<<=1) {
    int idx_to_add = tidx + idx_offset;
    if (idx_to_add < threads)
      sh_mem[tidx] += sh_mem[idx_to_add];
    __syncthreads();
  }

  return sh_mem[0];
}

__global__ void perform_w_update_buckets(DeviceData data, int subbucket) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

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

  const int kwxm = data.kernel_width_xm;
  const int num_evals_total = kwxm*kwxm;

  const float inducing_points_scale = 1.0 / data.inducing_point_step;
  const float kwx2 = data.kernel_width_xm/2.0;

  for (int data_idx = start; data_idx < end; ++data_idx) {

    Point x = data.mp[data_idx].p;
    float y = data.mp[data_idx].label;

    float wTphi = compute_wTphi(data, x, tidx, threads, phi_sparse);

    float c = -y * 1.0/(1.0 + __expf(y * wTphi));

    const int i0 = (x.x - data.opt.min) * inducing_points_scale - kwx2;
    const int j0 = (x.y - data.opt.min) * inducing_points_scale - kwx2;

    for (int eval_num=tidx; eval_num < num_evals_total; eval_num += threads) {
      int i = i0 + (eval_num / kwxm);
      int j = j0 + (eval_num % kwxm);

      // Evaluate kernel
      int idx_w = i*data.opt.inducing_points_n_dim + j;
      float x_m_x = i*data.inducing_point_step + data.opt.min;
      float x_m_y = j*data.inducing_point_step + data.opt.min;
      float k = kernel_lookup(data, x.x - x_m_x, x.y - x_m_y);
      if (i >=0 && i < data.opt.inducing_points_n_dim && j>=0 && j<data.opt.inducing_points_n_dim) {
        data.w[idx_w] -= data.opt.learning_rate * c * k;
      }
    }
  }
}

__global__ void compute_occupancy(DeviceData data, Point *points, float *res) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  extern __shared__ float phi_sparse[];

  Point x = points[bidx];

  float wTphi = compute_wTphi(data, x, tidx, threads, phi_sparse);

  if (tidx==0) {
    res[bidx] = 1.0 / (1.0 + __expf(-wTphi));
  }
}

std::vector<float> HilbertMap::GetOccupancy(std::vector<Point> points) {

  float *d_res;
  cudaMalloc(&d_res, sizeof(float)*points.size());

  Point *d_points;
  cudaMalloc(&d_points, sizeof(Point)*points.size());
  cudaMemcpy(d_points, points.data(), sizeof(Point)*points.size(), cudaMemcpyHostToDevice);

  int threads = 64;
  size_t blocks = points.size();
  int sm_size = threads*sizeof(float);
  //printf("Running with %dx%d threads and %d blocks and %ld sm_size\n",
  //    threads.x, threads.y, blocks, sm_size);
  compute_occupancy<<<blocks, threads, sm_size>>>(*data_, d_points, d_res);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }

  cudaFree(d_points);

  std::vector<float> res(points.size());
  cudaMemcpy(res.data(), d_res, sizeof(float)*points.size(), cudaMemcpyDeviceToHost);
  return res;
}

__global__ void compute_log_likelihood(DeviceData data, Point *points, float *gt_labels, float *scores) {

  // Figure out where this thread is
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  const int threads = blockDim.x;

  extern __shared__ float phi_sparse[];

  Point x = points[bidx];
  float y = gt_labels[bidx];

  float wTphi = compute_wTphi(data, x, tidx, threads, phi_sparse);

  if (threads==0) {
    scores[bidx] = log(1.0 + __expf(-y*wTphi));
  }
}

float HilbertMap::ComputeLogLikelihood(std::vector<Point> points, std::vector<float> gt_labels) {

  // Allocate
  float *d_scores;
  cudaMalloc(&d_scores, sizeof(float)*points.size());

  Point *d_points;
  cudaMalloc(&d_points, sizeof(Point)*points.size());
  cudaMemcpy(d_points, points.data(), sizeof(Point)*points.size(), cudaMemcpyHostToDevice);

  float *d_labels;
  cudaMalloc(&d_labels, sizeof(float)*gt_labels.size());
  cudaMemcpy(d_labels, gt_labels.data(), sizeof(float)*gt_labels.size(), cudaMemcpyHostToDevice);

  dim3 threads;
  threads.x = data_->kernel_width_xm;
  threads.y = data_->kernel_width_xm;
  threads.z = 1;
  size_t blocks = points.size();
  int sm_size = threads.x*threads.y*sizeof(float);
  compute_log_likelihood<<<blocks, threads, sm_size>>>(*data_, d_points, d_labels, d_scores);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Whoops\n");
  }

  thrust::device_ptr<float> dp_scores(d_scores);
  double score = thrust::reduce(dp_scores, dp_scores + points.size());
  cudaFree(d_scores);
  cudaFree(d_labels);
  cudaFree(d_points);

  return -score;
}

std::vector<float> HilbertMap::GetW() {
  int n_dim = data_->opt.inducing_points_n_dim;

  std::vector<float> w(n_dim*n_dim, 0.0f);

  cudaMemcpy(w.data(), data_->w, sizeof(float)*n_dim*n_dim, cudaMemcpyDeviceToHost);

  return w;
}

}
}
