#pragma once

#include <vector>

#include <Eigen/Core>

namespace library {
namespace bayesian_inference {

class IKernel {
 public:
  virtual double Compute(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const = 0;
};

class GaussianKernel : public IKernel {
 public:
  GaussianKernel(double r);

  virtual double Compute(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const;

 private:
  double radius_;
};

} // namespace bayesian_inference
} // namespace library
