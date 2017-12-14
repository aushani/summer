#include "library/bayesian_inference/kernel.h"

#include <boost/assert.hpp>

namespace library {
namespace bayesian_inference {

GaussianKernel::GaussianKernel(double r) :
 radius_(r) {
}

double GaussianKernel::Compute(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const {
  BOOST_ASSERT(sample.cols() == x_m.cols());
  BOOST_ASSERT(sample.rows() == 1);
  BOOST_ASSERT(x_m.rows() == 1);

  double sq_n = (x_m - sample).squaredNorm();
  double r2 = radius_*radius_;

  return std::exp(-sq_n / r2);
}

} // namespace bayesian_inference
} // namespace library
