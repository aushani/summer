#include "library/detector/device_model.cu.h"

namespace library {
namespace detector {

__global__ void UpdateModelKernel(DeviceModel model, const rt::DeviceOccGrid dog) {
  // Figure out which location this thread is processing
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (idx >= dog.size) {
    return;
  }

  rt::Location loc = dog.locs[idx];
  bool occ = dog.los[idx] > 0;

  // Update marginals
  int marginal_idx = model.GetIndex(loc);
  model.marginals[marginal_idx].IncrementCount(occ);
}

} // namespace detector
} // namespace library
