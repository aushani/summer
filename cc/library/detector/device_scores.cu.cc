#include "library/detector/device_scores.h"

namespace library {
namespace detector {

DeviceScores::DeviceScores(const Dim &d, float lp) : dim(d), log_prior(lp) {
  cudaMalloc(&d_scores, sizeof(float) * dim.Size());
  h_scores = (float*) malloc(sizeof(float) * dim.Size());
}

void DeviceScores::Cleanup() {
  cudaFree(d_scores);
  free(h_scores);
}

void DeviceScores::Reset() {
  cudaMemset(d_scores, 0, sizeof(float)*dim.Size());
  memset(h_scores, 0, sizeof(float)*dim.Size());
}

void DeviceScores::CopyToHost() {
  cudaMemcpy(h_scores, d_scores, sizeof(float)*dim.Size(), cudaMemcpyDeviceToHost);
}

float DeviceScores::GetScore(const ObjectState &os) const {
  int idx = dim.GetIndex(os);

  if (idx < 0) {
    return log_prior;
  }

  return h_scores[idx] + log_prior;
}

} // namespace ray_tracing
} // namespace library
