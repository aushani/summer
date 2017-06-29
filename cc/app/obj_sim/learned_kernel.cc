#include "app/obj_sim/learned_kernel.h"

#include <cmath>
#include <iostream>

LearnedKernel::LearnedKernel(float full_width, float res) :
  dim_size_(ceil(full_width/res)),
  res_(res),
  vals_(dim_size_*dim_size_, 0.0f) {
}

void LearnedKernel::CopyFrom(const hm::IKernel &kernel) {
  for (size_t i=0; i<dim_size_; i++) {
    for (size_t j=0; j<dim_size_; j++) {
      float dx = i*res_ - MaxSupport();
      float dy = j*res_ - MaxSupport();

      SetPixel(i, j, kernel.Evaluate(dx, dy));
    }
  }
}

float LearnedKernel::Evaluate(float dx, float dy) const {
  float x_k = dx + MaxSupport();
  float y_k = dy + MaxSupport();

  int i = round(x_k/res_);
  int j = round(y_k/res_);

  if (i<0 || j<0)
    return 0.0f;

  return GetPixel(i, j);
}

float LearnedKernel::MaxSupport() const {
  return res_*dim_size_/2.0f;
}

hm::DeviceKernelTable LearnedKernel::MakeDeviceKernelTable() const {
  hm::DeviceKernelTable kt(GetDimSize(), 2*MaxSupport());
  kt.SetData(vals_.data());
  kt.SetOrigin(-MaxSupport(), -MaxSupport());
  return kt;
}

float LearnedKernel::GetPixel(size_t i, size_t j) const {
  if (i>=dim_size_ || j>=dim_size_)
    return 0.0f;

  size_t idx = i*dim_size_ + j;

  return vals_[idx];
}

void LearnedKernel::SetPixel(size_t i, size_t j, float x) {
  if (i>=dim_size_ || j>=dim_size_)
    return;

  size_t idx = i*dim_size_ + j;

  vals_[idx] = x;
}

void LearnedKernel::SetLocation(float x, float y, float val) {
  float x_k = x + MaxSupport();
  float y_k = y + MaxSupport();

  int i = round(x_k/res_);
  int j = round(y_k/res_);

  if (i<0 || j<0) {
    return;
  }

  SetPixel(i, j, val);
}

size_t LearnedKernel::GetDimSize() const {
  return dim_size_;
}

float LearnedKernel::GetResolution() const {
  return res_;
}

const std::vector<float>& LearnedKernel::GetData() const {
  return vals_;
}
