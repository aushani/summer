#include "library/bayesian_inference/rvm.h"

#include <boost/assert.hpp>

namespace library {
namespace bayesian_inference {

Rvm::Rvm(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels) :
 training_data_(data),
 training_labels_(labels),
 x_m_(training_data_),
 w_(Eigen::VectorXd::Zero(data.size())) {
  BOOST_ASSERT(data.size() == labels.size());
}


} // namespace bayesian_inference
} // namespace library
