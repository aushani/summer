#include "library/ray_tracing/device_occ_grid.h"

namespace library {
namespace ray_tracing {

DeviceOccGrid::DeviceOccGrid(const Location *d_locs, const float *d_los, size_t sz, float r) :
 size(sz), res(r) {
  cudaMalloc(&locs, sizeof(Location) * size);
  cudaMalloc(&los, sizeof(float) * size);

  cudaMemcpy(locs, d_locs, sizeof(Location) * size, cudaMemcpyDeviceToDevice);
  cudaMemcpy(los, d_los, sizeof(float) * size, cudaMemcpyDeviceToDevice);
}

void DeviceOccGrid::Cleanup() {
  cudaFree(locs);
  cudaFree(los);
}

float DeviceOccGrid::GetResolution() const {
  return res;
}

}  // namespace ray_tracing
}  // namespace library

