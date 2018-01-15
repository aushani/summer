#include "library/bayesian_inference/rvm.h"

#include <ceres/ceres.h>
#include <boost/assert.hpp>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>

#include "library/timer/timer.h"

namespace tr = library::timer;

namespace library {
namespace bayesian_inference {

class RvmWeightFunction : public ceres::FirstOrderFunction {
 public:
  RvmWeightFunction(const Rvm &rvm) : rvm_(rvm) {}

  virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const {
    Eigen::MatrixXd w(rvm_.NumRelevanceVectors(), 1);
    for (int i=0; i<w.rows(); i++) {
      w(i, 0) = parameters[i];
    }

    Eigen::MatrixXd gradient_analytical;

    double ll = rvm_.ComputeLogLikelihood(w, &gradient_analytical);
    cost[0] = -ll;

    if (gradient != nullptr) {
      //double step = 1e-6;
      for (int i=0; i<w.rows(); i++) {
        //Eigen::MatrixXd w1 = w;
        //w1(i, 0) += step;

        //double ll1 = rvm_.ComputeLogLikelihood(w1);
        //gradient[i] = -(ll1 - ll) / (step);
        //printf("grad %d = %f vs %f\n", i, gradient[i], -gradient_analytical(i, 0));
        gradient[i] = -gradient_analytical(i, 0);
      }
    }

    return true;
  }

  virtual int NumParameters() const {
    return rvm_.NumRelevanceVectors();
  }

 private:
  const Rvm &rvm_;

};


Rvm::Rvm(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels) :
 training_data_(data),
 training_labels_(labels),
 x_m_(training_data_),
 w_(Eigen::MatrixXd::Zero(data.rows(), 1)),
 alpha_(Eigen::MatrixXd::Ones(data.rows(), 1)) {
  BOOST_ASSERT(data.rows() == labels.rows());

  phi_samples_ = ComputePhi(training_data_);

  printf("Training data is %ld x %ld\n", training_data_.rows(), training_data_.cols());
  printf("Training labels is %ld x %ld\n", training_labels_.rows(), training_labels_.cols());
  printf("x_m is %ld x %ld\n", x_m_.rows(), x_m_.cols());
  printf("w is %ld x %ld\n", w_.rows(), w_.cols());
  printf("phi is %ld x %ld\n", phi_samples_.rows(), phi_samples_.cols());
}

double Rvm::ComputeBasisFunction(const Eigen::MatrixXd &sample, const Eigen::MatrixXd &x_m) const {
  BOOST_ASSERT(sample.cols() == x_m.cols());
  BOOST_ASSERT(sample.rows() == 1);
  BOOST_ASSERT(x_m.rows() == 1);

  double sq_n = (x_m - sample).squaredNorm();
  double r2 = kBasisFunctionR_ * kBasisFunctionR_;

  return std::exp(-sq_n / r2);
}

Eigen::SparseMatrix<double> Rvm::ComputePhi(const Eigen::MatrixXd &data) const {

  //Eigen::MatrixXd phi(data.rows(), x_m_.rows());
  Eigen::SparseMatrix<double> phi;
  phi.conservativeResize(data.rows(), x_m_.rows());

  for (int i=0; i<data.rows(); i++) {
    auto sample = data.row(i);
    for (int j=0; j<x_m_.rows(); j++){
      auto x_m = x_m_.row(j);

      //phi.insert(i, j) = ComputeBasisFunction(sample, x_m);

      // Sparsify
      double val = ComputeBasisFunction(sample, x_m);
      if (std::abs(val) > 1e-3) {
        phi.insert(i, j) = val;
      }
    }
  }

  return phi;
}

const Eigen::MatrixXd& Rvm::GetRelevanceVectors() const {
  return x_m_;
}

int Rvm::NumRelevanceVectors() const {
  return x_m_.rows();
}

Eigen::MatrixXd Rvm::PredictLabels(const Eigen::MatrixXd &samples) const {
  Eigen::MatrixXd phi = ComputePhi(samples);

  Eigen::MatrixXd exponent = -phi * w_;
  auto exp = exponent.array().exp();
  auto prob = (1 + exp).inverse();

  return prob;
}

double Rvm::ComputeLogLikelihood(const Eigen::MatrixXd &w, Eigen::MatrixXd *gradient) const {
  Eigen::MatrixXd exponent = -phi_samples_ * w;
  auto exp = exponent.array().exp();
  Eigen::ArrayXXd y_n = (1 + exp).inverse();

  Eigen::ArrayXXd t_n = training_labels_.array();

  double c1 = (t_n * y_n.log()).sum();
  double c2 = ((1 - t_n) * (1 - y_n).log()).sum();

  Eigen::MatrixXd A = alpha_.col(0).asDiagonal();
  auto prod = -0.5 * w.transpose() * A * w;
  BOOST_ASSERT(prod.rows() == 1 && prod.cols() == 1);
  double c3 = prod(0, 0);

  if (gradient != nullptr) {
    //printf("getting grad...\n");
    //printf("\tx_m is %ld x %ld\n", x_m_.rows(), x_m_.cols());
    //printf("\tw is %ld x %ld\n", w.rows(), w.cols());
    //printf("\tphi is %ld x %ld\n", phi_samples_.rows(), phi_samples_.cols());
    //printf("\tt_n is %ld x %ld\n", t_n.rows(), t_n.cols());
    //printf("\ty_n is %ld x %ld\n", y_n.rows(), y_n.cols());

    //tr::Timer t;
    (*gradient) = phi_samples_.transpose() * (t_n - y_n).matrix() - A*w;
    //printf("got grad in %5.3f ms\n", t.GetMs());
  }

  return c1 + c2 + c3;
}

void Rvm::Solve(int iterations) {
  for (int i=0; i<iterations; i++) {
    tr::Timer t;
    tr::Timer t_step;

    printf("Iteration %d / %d\n", i, iterations);

    t_step.Start();
    bool res = UpdateW();
    printf("\tUpdate w took %5.3f ms, LL now %f\n", t_step.GetMs(), ComputeLogLikelihood(w_));

    if (!res) {
      printf("\tUpdate w not successful, aborting\n");
      return;
    }

    t_step.Start();
    UpdateAlpha();
    printf("\tUpdate alpha took %5.3f ms\n", t_step.GetMs());

    // Prune parameters
    t_step.Start();
    PruneXm();
    printf("\tPruning took %5.3f ms, have %d relevant vectors\n", t_step.GetMs(), NumRelevanceVectors());

    //printf("\tx_m is %ld x %ld\n", x_m_.rows(), x_m_.cols());
    //printf("\tw is %ld x %ld\n", w_.rows(), w_.cols());
    //printf("\tphi is %ld x %ld\n", phi_samples_.rows(), phi_samples_.cols());

    //printf("\nIteration took %5.3f ms\n", t.GetMs());
  }
}

bool Rvm::UpdateW() {
  ceres::GradientProblem problem(new RvmWeightFunction(*this));
  ceres::GradientProblemSolver::Options options;
  //options.minimizer_progress_to_stdout = true;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 100;
  options.max_num_line_search_step_size_iterations = 100;
  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(options, problem, w_.data(), &summary);

  //std::cout << summary.FullReport() << std::endl;
  return summary.IsSolutionUsable();
}

void Rvm::UpdateAlpha() {
  //tr::Timer t;

  //t.Start();
  Eigen::MatrixXd exponent = -phi_samples_ * w_;
  auto exp = exponent.array().exp();
  Eigen::ArrayXXd y_n = (1 + exp).inverse();

  Eigen::ArrayXXd b = y_n * (1 - y_n);
  BOOST_ASSERT(b.cols() == 1);
  //printf("Took %5.3f ms to compute\n", t.GetMs());

  //t.Start();
  //Eigen::MatrixXd b_mat = b.matrix().col(0).asDiagonal();
  Eigen::SparseMatrix<double> b_mat;
  b_mat.conservativeResize(b.rows(), b.rows());
  for (int i=0; i<b.rows(); i++) {
    b_mat.insert(i, i) = b(i, 0);
  }

  //Eigen::MatrixXd A = alpha_.col(0).asDiagonal();
  Eigen::SparseMatrix<double> A;
  A.conservativeResize(alpha_.rows(), alpha_.rows());
  for (int i=0; i<alpha_.rows(); i++) {
    A.insert(i, i) = alpha_(i);
  }

  //Eigen::SparseMatrix<double> h_neg = (phi_samples_.transpose() * b_mat * phi_samples_ + A).eval();
  Eigen::SparseMatrix<double> h_neg = (phi_samples_.transpose() * b_mat * phi_samples_ + A);
  Eigen::MatrixXd identity(h_neg.rows(), h_neg.cols());
  identity.setIdentity();
  //printf("Took %5.3f ms to build problem\n", t.GetMs());

  //t.Start();
  //Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, NaturalOrdering<int> > llt(h_neg);
  //Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int> > llt(h_neg);
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > llt(h_neg);
  Eigen::MatrixXd cov = llt.solve(identity);
  //printf("Took %5.3f ms to invert %ld x %ld\n", t.GetMs(), cov.rows(), cov.cols());

  //t.Start();
  int n_x_m = NumRelevanceVectors();
  for (int i=0; i<n_x_m; i++) {
    double w2 = w_(i, 0) * w_(i, 0);
    alpha_(i, 0) = (1 - alpha_(i, 0) * cov(i, i)) / w2;
  }
  //printf("Took %5.3f ms to update\n", t.GetMs());
}

// from https://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
void Rvm::RemoveRow(Eigen::MatrixXd *matrix, unsigned int rowToRemove) {
  unsigned int numRows = matrix->rows()-1;
  unsigned int numCols = matrix->cols();

  if (rowToRemove < numRows) {
    matrix->block(rowToRemove,0,numRows-rowToRemove,numCols) =
      matrix->block(rowToRemove+1,0,numRows-rowToRemove,numCols).eval();
  }

  matrix->conservativeResize(numRows,numCols);
}

void Rvm::RemoveColumn(Eigen::MatrixXd *matrix, unsigned int colToRemove) {
  unsigned int numRows = matrix->rows();
  unsigned int numCols = matrix->cols()-1;

  if (colToRemove < numCols) {
    matrix->block(0,colToRemove,numRows,numCols-colToRemove) =
      matrix->block(0,colToRemove+1,numRows,numCols-colToRemove).eval();
  }

  matrix->conservativeResize(numRows,numCols);
}

void Rvm::PruneXm() {
  const double cutoff = 1e3;

  int n_x_m = NumRelevanceVectors();

  // Walk backwards
  int x_m_at = n_x_m - 1;

  while (x_m_at >= 0) {
    if (alpha_(x_m_at, 0) > cutoff) {
      // Prune
      //RemoveColumn(&phi_samples_, x_m_at);
      RemoveRow(&x_m_, x_m_at);
      RemoveRow(&w_, x_m_at);
      RemoveRow(&alpha_, x_m_at);
    }

    x_m_at--;
  }

  // Rebuild phi_samples_
  phi_samples_ = ComputePhi(training_data_);
}


} // namespace bayesian_inference
} // namespace library
