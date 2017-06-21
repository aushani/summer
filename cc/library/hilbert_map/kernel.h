#pragma once

namespace library {
namespace hilbert_map {

class IKernel {
 public:
  virtual ~IKernel() {};

  virtual float Evaluate(float dx, float dy) const = 0;
  virtual float MaxSupport() const = 0;
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
