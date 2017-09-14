#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "library/detector/object_state.h"
#include "library/detector/dim.h"

namespace library {
namespace detector {

struct DeviceScores {
  float *h_scores = nullptr;
  float *d_scores = nullptr;

  Dim dim;
  float log_prior = 0;

  DeviceScores(const Dim &d, float log_prior);

  void Cleanup();

  void Reset();
  void CopyToHost();

  float GetScore(const ObjectState &os) const;
};

} // namespace ray_tracing
} // namespace library
