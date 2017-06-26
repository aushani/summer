#pragma once

#include <vector>
#include <cstdlib>

#include "library/hilbert_map/kernel.h"

namespace hm = library::hilbert_map;

class LearnedKernel : public hm::IKernel {
 public:
  LearnedKernel(float size, float res);

  void CopyFrom(const hm::IKernel &kernel);

  float Evaluate(float dx, float dy) const override;
  float MaxSupport() const override;

  hm::DeviceKernelTable MakeDeviceKernelTable() const override;

  float GetPixel(int i, int j) const;
  void SetPixel(int i, int j, float x);

  size_t GetDimSize() const;
  float GetResolution() const;

 private:
  size_t dim_size_;
  float res_;

  std::vector<float> vals_;

};
