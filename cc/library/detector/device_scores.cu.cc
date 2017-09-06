#include "device_scores.h"

namespace library {
namespace detector {

DeviceScores::DeviceScores(float r, float range_x, float range_y, float lp)  :
 n_x(2 * ceil(range_x / r) + 1),
 n_y(2 * ceil(range_y / r) + 1),
 res(r),
 log_prior(lp) {
  cudaMalloc(&d_scores, sizeof(float)*n_x*n_y);
  h_scores = (float*) malloc(sizeof(float) * n_x * n_y);
}

void DeviceScores::Cleanup() {
  cudaFree(d_scores);
  free(h_scores);
}

void DeviceScores::Reset() {
  cudaMemset(d_scores, 0, sizeof(float)*n_x*n_y);
  memset(h_scores, 0, sizeof(float)*n_x*n_y);
}

void DeviceScores::CopyToHost() {
  cudaMemcpy(h_scores, d_scores, sizeof(float)*n_x*n_y, cudaMemcpyDeviceToHost);
}

float DeviceScores::GetScore(const ObjectState &os) const {
  int idx = GetIndex(os);

  if (idx < 0) {
    return log_prior;
  }

  return h_scores[idx] + log_prior;
}

} // namespace ray_tracing
} // namespace library
