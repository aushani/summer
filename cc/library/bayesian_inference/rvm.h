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

  const Eigen::MatrixXd& GetRelevanceVectors() const;
  int NumRelevanceVectors() const;

  Eigen::MatrixXd PredictLabels(const Eigen::MatrixXd &samples) const;

  double ComputeLogLikelihood(const Eigen::MatrixXd &w) const;

  void Solve(int iterations);

 private:
  const double kBasisFunctionR_ = 0.5;

  Eigen::MatrixXd training_data_;
  Eigen::MatrixXd training_labels_;

  Eigen::MatrixXd x_m_;
  Eigen::MatrixXd w_;
  Eigen::VectorXd alpha_;

  Eigen::MatrixXd phi_samples_;

  double ComputeBasisFunction(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const;
  Eigen::MatrixXd ComputePhi(const Eigen::MatrixXd &data) const;

  void UpdateW();
  void UpdateAlpha();
};

} // namespace bayesian_inference
} // namespace library
