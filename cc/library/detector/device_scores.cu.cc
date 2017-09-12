#include "library/detector/device_scores.h"

#include <iostream>

#include <boost/assert.hpp>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace library {
namespace detector {

DeviceScores::DeviceScores(float r, float range_x, float range_y, float lp, int threads)  :
 n_x(2 * ceil(range_x / r) + 1),
 n_y(2 * ceil(range_y / r) + 1),
 num_pos(n_x * n_y),
 res(r),
 log_prior(lp),
 scoring_threads(threads) {

  cudaMalloc(&d_scores, sizeof(float)*num_pos);
  h_scores = (float*) malloc(sizeof(float) * num_pos);

  cudaError_t err = cudaMalloc(&d_scores_thread, sizeof(float)*num_pos * scoring_threads);
  BOOST_ASSERT(err == cudaSuccess);

  printf("Allocated %ld MBytes for scores\n", sizeof(float) * num_pos * (scoring_threads + 1) / (1024*124));
}

void DeviceScores::Cleanup() {
  cudaFree(d_scores);
  free(h_scores);
}

void DeviceScores::Reset() {
  cudaMemset(d_scores, 0, sizeof(float)*num_pos);
  memset(h_scores, 0, sizeof(float)*num_pos);

  thrust::device_ptr<float> dp_st(d_scores_thread);
  thrust::fill(dp_st, dp_st + num_pos * scoring_threads, 0.0f);
}

void DeviceScores::CopyToHost() {
  cudaMemcpy(h_scores, d_scores, sizeof(float)*num_pos, cudaMemcpyDeviceToHost);
}

float DeviceScores::GetScore(const ObjectState &os) const {
  int idx = GetIndex(os);

  if (idx < 0) {
    return log_prior;
  }

  return h_scores[idx] + log_prior;
}

__global__ void ReduceKernel(DeviceScores scores) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= scores.num_pos) {
    return;
  }

  float score = 0;
  for (int i=0; i<scores.scoring_threads; i++) {
    score += scores.d_scores_thread[idx + i*scores.num_pos];
  }

  scores.d_scores[idx] = score;
}

void DeviceScores::Reduce() {
  int threads = 1024;
  int blocks = std::ceil(num_pos / static_cast<double>(threads));

  //printf("\t\t\tReducing %d scores with %d threads and %d blocks\n", num_pos, threads, blocks);
  ReduceKernel<<<blocks, threads>>>(*this);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);
}

} // namespace ray_tracing
} // namespace library
