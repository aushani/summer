#pragma once

// This is gross but lets this play nicely with both cuda and non-cuda compilers
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "library/ray_tracing/stats.h"

namespace library {
namespace ray_tracing {

struct Feature {
  float theta = 0;
  float phi = 0;
  float intensity = 0;

  CUDA_CALLABLE Feature() {}

  CUDA_CALLABLE Feature(float t, float p, float i) :
    theta(t), phi(p), intensity(i) {}

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & theta;
    ar & phi;
  }
};

}  // namespace ray_tracing
}  // namespace library
