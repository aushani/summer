#include "library/hilbert_map/kernel.h"

#include <vector>

#include <iostream>

namespace library {
namespace hilbert_map {

DeviceKernelTable::DeviceKernelTable(int nd, float full_width) :
 n_dim(nd),
 resolution(full_width/n_dim),
 scale(1.0f/resolution) {
  cudaMalloc(&kernel_table, sizeof(float)*n_dim*n_dim);
}

DeviceKernelTable::~DeviceKernelTable() {

}

void DeviceKernelTable::Cleanup() {
  if (kernel_table) {
    cudaFree(kernel_table);
    kernel_table = NULL;
  }
}

void DeviceKernelTable::SetData(const float *data) {
  cudaMemcpy(kernel_table, data, sizeof(float)*n_dim*n_dim, cudaMemcpyHostToDevice);
}

void DeviceKernelTable::SetOrigin(float x, float y) {
  x0 = x;
  y0 = y;
}

DeviceKernelTable IKernel::MakeDeviceKernelTable() const {

  DeviceKernelTable kt(1024, 2*MaxSupport());

  std::vector<float> kt_host;
  for (int i=0; i<kt.n_dim; i++) {
    float dx = i * 2.0 * MaxSupport() / kt.n_dim - MaxSupport();
    for (int j=0; j<kt.n_dim; j++) {
      float dy = j * 2.0 * MaxSupport() / kt.n_dim - MaxSupport();
      kt_host.push_back(Evaluate(dx, dy));
    }
  }

  kt.SetData(kt_host.data());
  kt.SetOrigin(-MaxSupport(), -MaxSupport());

  return kt;
}

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
