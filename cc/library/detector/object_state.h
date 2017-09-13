#pragma once

// This is gross but lets this play nicely with both cuda and non-cuda compilers
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace library {
namespace detector {

struct ObjectState {
  float x;
  float y;

  int angle_bin;

  CUDA_CALLABLE ObjectState(float xx, float yy, int b) :
    x(xx), y(yy), angle_bin(b) {}

  CUDA_CALLABLE bool operator<(const ObjectState &rhs) const {
    if (fabs(x - rhs.x) >= kTolerance_) {
      return x < rhs.x;
    }

    if (fabs(y - rhs.y) >= kTolerance_) {
      return y < rhs.y;
    }

    return angle_bin < rhs.angle_bin;
  }

 private:
  static constexpr float kTolerance_ = 0.001;
};

} // namespace detector
} // namespace library
