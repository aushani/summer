#include "library/bayesian_inference/rvm.h"

#include <boost/assert.hpp>

namespace library {
namespace bayesian_inference {

Rvm::Rvm(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels) :
 training_data_(data),
 training_labels_(labels),
 x_m_(training_data_),
 w_(Eigen::MatrixXd::Zero(data.size(), 1)) {
  BOOST_ASSERT(data.rows() == labels.rows());

  printf("Training data is %ld x %ld\n", training_data_.rows(), training_data_.cols());
  printf("Training labels is %ld x %ld\n", training_labels_.rows(), training_labels_.cols());
  printf("x_m is %ld x %ld\n", x_m_.rows(), x_m_.cols());
  printf("w is %ld x %ld\n", w_.rows(), w_.cols());

  phi_samples_ = ComputePhi(data);
}

double Rvm::ComputeBasisFunction(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const {
  BOOST_ASSERT(sample.cols() == x_m.cols());
  BOOST_ASSERT(sample.rows() == 1);
  BOOST_ASSERT(x_m.rows() == 1);

  double sq_n = (x_m - sample).squaredNorm();
  double r2 = kBasisFunctionR_ * kBasisFunctionR_;

  return std::exp(-sq_n / r2);
}

Eigen::MatrixXd Rvm::ComputePhi(const Eigen::MatrixXd &data) const {

  Eigen::MatrixXd phi(data.rows(), x_m_.rows());

  for (int i=0; i<data.rows(); i++) {
    auto sample = data.row(i);
    for (int j=0; j<x_m_.rows(); j++){
      auto x_m = x_m_.row(i);

      phi(i, j) = ComputeBasisFunction(sample, x_m);
    }
  }

  return phi;
}


} // namespace bayesian_inference
} // namespace library
