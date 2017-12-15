#pragma once

#include <vector>

#include <boost/assert.hpp>

#include <Eigen/Core>

namespace library {
namespace bayesian_inference {

template <class T>
class IKernel {
 public:
  virtual double Compute(const T &sample, const T &x_m) const = 0;
};

class GaussianKernel : public IKernel<Eigen::VectorXd> {
 public:
  GaussianKernel(double r) : radius_(r) {}

  virtual double Compute(const Eigen::VectorXd &sample, const Eigen::VectorXd &x_m) const {
    BOOST_ASSERT(sample.size() == x_m.size());

    double sq_n = (x_m - sample).squaredNorm();
    double r2 = radius_*radius_;

    return std::exp(-sq_n / r2);
  }

 private:
  double radius_;
};

} // namespace bayesian_inference
} // namespace library
