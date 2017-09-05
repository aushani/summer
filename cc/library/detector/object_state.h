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

  float theta;

  CUDA_CALLABLE ObjectState(float xx, float yy, float tt) :
    x(xx), y(yy), theta(tt) {}

  CUDA_CALLABLE bool operator<(const ObjectState &rhs) const {
    if (fabs(x - rhs.x) >= kTolerance_) {
      return x < rhs.x;
    }

    if (fabs(y - rhs.y) >= kTolerance_) {
      return y < rhs.y;
    }

    if (fabs(theta - rhs.theta) >= kTolerance_) {
      return theta < rhs.theta;
    }

    return false;
  }

 private:
  static constexpr float kTolerance_ = 0.001;
};

} // namespace detector
} // namespace library
