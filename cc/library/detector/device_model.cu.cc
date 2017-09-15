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

  // Update conditionals
  if (model.conditionals != nullptr) {
    for (int i_given=0; i_given<dog.size; i_given++) {
      rt::Location loc_given = dog.locs[i_given];
      int conditional_idx = model.GetIndex(loc, loc_given);
      bool occ_given = dog.los[i_given] > 0;

      model.conditionals[conditional_idx].IncrementCount(occ, occ_given);
    }
  }
}

} // namespace detector
} // namespace library
