#include "library/gpu_util/util.h"

#include <boost/assert.hpp>

namespace library {
namespace gpu_util {

bool SetDevice(int device) {
  cudaError_t err = cudaSetDevice(device);
  BOOST_ASSERT(err == cudaSuccess);

  return true;
}

}
}
