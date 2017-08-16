#pragma once

// This is gross but lets this play nicely with both cuda and non-cuda compilers
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace library {
namespace ray_tracing {

// This struct represents a location in an occupancy grid, providing a
// discrete index given by (i, j, k). Indicies could be negative.
struct Location {
  int i = 0;
  int j = 0;
  int k = 0;

  CUDA_CALLABLE Location() { }

  CUDA_CALLABLE Location(int ii, int jj, int kk) :
    i(ii), j(jj), k(kk) { }

  CUDA_CALLABLE Location(float x, float y, float z, float res) :
    i(round(x / res)), j(round(y / res)), k(round(z / res)) { }

  CUDA_CALLABLE bool operator<(const Location &rhs) const {
    if (i != rhs.i) {
      return i < rhs.i;
    }
    if (j != rhs.j) {
      return j < rhs.j;
    }
    return k < rhs.k;
  }

  CUDA_CALLABLE bool operator==(const Location &rhs) const {
    return i == rhs.i && j == rhs.j && k == rhs.k;
  }
};

}  // namespace ray_tracing
}  // namespace library
