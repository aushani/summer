#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace library {
namespace detector {

struct Dim {
  float min_x;
  float max_x;

  float min_y;
  float max_y;

  float res;

  float range_x;
  float range_y;

  int n_x;
  int n_y;
  int size;

  Dim(float x0, float x1, float y0, float y1, float r) :
   min_x(x0),
   max_x(x1),
   min_y(y0),
   max_y(y1),
   res(r),
   range_x(max_x - min_x), range_y(max_y - min_y),
   n_x(std::ceil(range_x / res)),
   n_y(std::ceil(range_y / res)),
   size(n_x * n_y) {}

  CUDA_CALLABLE ObjectState GetState(int idx) const {
    int ix = idx / n_y;
    int iy = idx % n_y;

    return GetState(ix, iy);
  }

  CUDA_CALLABLE ObjectState GetState(int ix, int iy) const {
    float x = ix * res + res/2 + min_x;
    float y = iy * res + res/2 + min_y;

    return ObjectState(x, y, 0);
  }

  CUDA_CALLABLE int GetIndex(const ObjectState &os) const {
    return GetIndex(os.x, os.y);
  }

  CUDA_CALLABLE int GetIndex(double x, double y) const {
    int ix = (x - min_x) / res;
    int iy = (y - min_y) / res;

    if (ix >= n_x || iy >= n_y || ix < 0 || iy < 0) {
      return -1;
    }

    size_t idx = ix * n_y + iy;
    return idx;
  }

  CUDA_CALLABLE bool InRange(int idx) const {
    return idx >= 0 && idx < Size();
  }

  CUDA_CALLABLE bool InRange(const ObjectState &os) const {
    return InRange(os.x, os.y);
  }

  CUDA_CALLABLE bool InRange(double x, double y) const {
    return x >= min_x && x < max_x && y >= min_y && y < max_y;
  }

  CUDA_CALLABLE int Size() const {
    return size;
  }
};

} // namespace detector
} // namespace library
