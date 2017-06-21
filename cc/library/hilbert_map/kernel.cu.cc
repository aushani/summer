#include "kernel.h"

namespace library {
namespace hilbert_map {

SparseKernel::SparseKernel(float kwm) :
  kernel_width_meters_(kwm) {

}

float SparseKernel::Evaluate(float dx, float dy) const {
  float d2 = dx*dx + dy*dy;
  float kernel_width_meters_sq = kernel_width_meters_*kernel_width_meters_;

  if (d2 > kernel_width_meters_sq)
    return 0;

  float r = sqrt(d2);

  // Apply kernel width
  r /= kernel_width_meters_;

  float t = 2 * M_PI * r;

  return (2 + cosf(t)) / 3 * (1 - r) + 1.0/(2 * M_PI) * sinf(t);
}

float SparseKernel::MaxSupport() const {
  return kernel_width_meters_;
}

BoxKernel::BoxKernel(float kwm) :
  kernel_width_meters_(kwm) {

}

float BoxKernel::Evaluate(float dx, float dy) const {
  if (std::abs(dx) < kernel_width_meters_ && std::abs(dy) < kernel_width_meters_)
    return 1.0f;

  return 0.0f;
}

float BoxKernel::MaxSupport() const {
  return kernel_width_meters_;
}

}
}
