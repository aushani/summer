#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "library/detector/object_state.h"

namespace library {
namespace detector {

struct DeviceScores {
  float *h_scores = nullptr;
  float *d_scores = nullptr;

  int n_x = 0;
  int n_y = 0;
  size_t num_pos = 0;
  float res = 0;

  float log_prior = 0;

  float *d_scores_thread = nullptr;
  int scoring_threads = 0;

  DeviceScores(float r, float range_x, float range_y, float log_prior, int threads);

  void Cleanup();

  void Reset();
  void CopyToHost();

  float GetScore(const ObjectState &os) const;

  CUDA_CALLABLE bool InRange(int idx) const {
    return idx >= 0 && idx < Size();
  }

  CUDA_CALLABLE bool InRange(const ObjectState &os) const {
    return InRange(GetIndex(os));
  }

  CUDA_CALLABLE int Size() const {
    return n_x * n_y;
  }

  CUDA_CALLABLE ObjectState GetState(int idx) const {
    int ix = idx / n_y;
    int iy = idx % n_y;

    // int instead of size_t because could be negative
    int dix = ix - n_x/2;
    int diy = iy - n_y/2;

    float x = dix * res + res/2;
    float y = diy * res + res/2;

    return ObjectState(x, y, 0);
  }

  CUDA_CALLABLE int GetIndex(const ObjectState &os) const {
    int ix = os.x / res + n_x / 2;
    int iy = os.y / res + n_y / 2;

    if (ix >= n_x || iy >= n_y) {
      return -1;
    }

    size_t idx = ix * n_y + iy;
    return idx;
  }

  CUDA_CALLABLE int GetThreadIndex(const ObjectState &os, int thread) const {
    if (thread < 0 || thread >= scoring_threads) {
      return -1;
    }

    size_t idx = GetIndex(os);
    return thread*num_pos + idx;
  }

  void Reduce();
};

} // namespace ray_tracing
} // namespace library
