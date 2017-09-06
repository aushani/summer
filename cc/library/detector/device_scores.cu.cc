#include "device_scores.h"

namespace library {
namespace detector {

DeviceScores::DeviceScores(float r, double range_x, double range_y)  :
   n_x(2 * ceil(range_x / r) + 1),
   n_y(2 * ceil(range_y / r) + 1),
   res(r) {
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
    return 0.0;
  }

  return h_scores[idx];
}

} // namespace ray_tracing
} // namespace library
