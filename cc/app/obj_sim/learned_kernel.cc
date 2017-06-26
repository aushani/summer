#include "app/obj_sim/learned_kernel.h"

#include <cmath>
#include <iostream>

LearnedKernel::LearnedKernel(float size, float res) :
  dim_size_(ceil(size/res)),
  res_(res),
  vals_(dim_size_*dim_size_, 0.0f) {
}

void LearnedKernel::CopyFrom(const hm::IKernel &kernel) {
  for (int i=0; i<dim_size_; i++) {
    for (int j=0; j<dim_size_; j++) {
      float dx = i*res_;
      float dy = j*res_;

      SetPixel(i, j, kernel.Evaluate(dx, dy));
    }
  }
}

float LearnedKernel::Evaluate(float dx, float dy) const {
  int i = round(std::fabs(dx)/res_);
  int j = round(std::fabs(dy)/res_);

  return GetPixel(i, j);
}

float LearnedKernel::MaxSupport() const {
  return res_*dim_size_;
}

hm::DeviceKernelTable LearnedKernel::MakeDeviceKernelTable() const {
  hm::DeviceKernelTable kt(GetDimSize(), MaxSupport());
  kt.SetData(vals_.data());
  return kt;
}

float LearnedKernel::GetPixel(int i, int j) const {
  if (i<0 || i>=dim_size_ || j<0 || j>=dim_size_)
    return 0.0;

  size_t idx = i*dim_size_ + j;

  return vals_[idx];
}

void LearnedKernel::SetPixel(int i, int j, float x) {
  size_t idx = i*dim_size_ + j;

  vals_[idx] = x;
}

size_t LearnedKernel::GetDimSize() const {
  return dim_size_;
}

float LearnedKernel::GetResolution() const {
  return res_;
}
