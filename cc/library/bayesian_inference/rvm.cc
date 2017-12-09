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

const Eigen::MatrixXd& Rvm::GetRelevanceVectors() const {
  return x_m_;
}

Eigen::MatrixXd Rvm::PredictLabels(const Eigen::MatrixXd &samples) const {
  Eigen::MatrixXd phi = ComputePhi(samples);

  Eigen::MatrixXd exponent = -phi * w_;
  auto exp = exponent.array().exp();
  auto prob = (1 + exp).inverse();

  return prob;
}

double Rvm::ComputeLogLikelihood(const Eigen::MatrixXd &w) const {
  Eigen::MatrixXd exponent = -phi_samples_ * w_;
  auto exp = exponent.array().exp();
  Eigen::ArrayXXd y_n = (1 + exp).inverse();

  Eigen::ArrayXXd t_n = training_labels_.array();

  double c1 = (t_n * y_n.log()).sum();
  double c2 = ((1 - t_n) * (1 - y_n).log()).sum();

  Eigen::MatrixXd A = alpha_.asDiagonal();
  auto prod = -0.5 * w_.transpose() * A * w_;
  BOOST_ASSERT(prod.rows() == 0 && prod.cols() == 0);
  double c3 = prod(0, 0);

  return c1 + c2 + c3;
}


} // namespace bayesian_inference
} // namespace library
