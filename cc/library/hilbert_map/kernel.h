#pragma once

#include <cuda_runtime.h>

namespace library {
namespace hilbert_map {

struct DeviceKernelTable {
  float *kernel_table = NULL;

  int n_dim = 0;
  float resolution = 0.0f;
  float scale = 0.0f;

  DeviceKernelTable(int nd, float max_support);
  ~DeviceKernelTable();

  void Cleanup();

  void SetData(const float *data);
};

class IKernel {
 public:
  virtual ~IKernel() {};

  virtual float Evaluate(float dx, float dy) const = 0;
  virtual float MaxSupport() const = 0;

  virtual DeviceKernelTable MakeDeviceKernelTable() const;
};

class SparseKernel : public IKernel {
 public:
  SparseKernel(float kwm);

  float Evaluate(float dx, float dy) const override;
  float MaxSupport() const override;

 private:
  float kernel_width_meters_;
};

class BoxKernel : public IKernel {
 public:
  BoxKernel(float kwm);

  float Evaluate(float dx, float dy) const override;
  float MaxSupport() const override;

 private:
  float kernel_width_meters_;
};

}
}
