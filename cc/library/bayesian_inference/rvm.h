#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

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

  double ComputeLogLikelihood(const Eigen::MatrixXd &w, Eigen::MatrixXd *gradient = nullptr) const;

  void Solve(int iterations);

 private:
  const double kBasisFunctionR_ = 0.5;

  Eigen::MatrixXd training_data_;
  Eigen::MatrixXd training_labels_;

  Eigen::MatrixXd x_m_;
  Eigen::MatrixXd w_;
  Eigen::MatrixXd alpha_;

  Eigen::SparseMatrix<double> phi_samples_;

  double ComputeBasisFunction(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const;
  Eigen::SparseMatrix<double> ComputePhi(const Eigen::MatrixXd &data) const;

  bool UpdateW();
  void UpdateAlpha();
  void PruneXm();

  static void RemoveRow(Eigen::MatrixXd *matrix, unsigned int rowToRemove);
  static void RemoveColumn(Eigen::MatrixXd *matrix, unsigned int rowToRemove);
};

} // namespace bayesian_inference
} // namespace library
