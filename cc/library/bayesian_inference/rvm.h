#pragma once

#include <vector>

#include <Eigen/Core>

namespace library {
namespace bayesian_inference {

class Rvm {
 public:
  // data is n_samples X dimension
  // labels is n_samples X 1
  Rvm(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels);

 private:
  Eigen::MatrixXd training_data_;
  Eigen::MatrixXd training_labels_;

  Eigen::MatrixXd x_m_;
  Eigen::MatrixXd w_;
};

} // namespace bayesian_inference
} // namespace library
