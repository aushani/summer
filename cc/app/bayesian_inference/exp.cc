#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>

#include <Eigen/Core>

#include "library/bayesian_inference/rvm.h"

namespace bi = library::bayesian_inference;

int main(int argc, char **argv) {
  printf("RVM Test\n");

  int n_samples = 100;
  int dim = 2;

  Eigen::MatrixXd data(n_samples, dim);
  Eigen::MatrixXd labels(n_samples, 1);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  // Generate samples
  std::normal_distribution<double> distribution(0, 0.25);

  for (int i=0; i<n_samples; i++) {
    for (int j=0; j<dim; j++) {
      data(i, j) = distribution(rand_engine);
    }

    double x = data(i, 0);
    double y = data(i, 1);
    double val = 0.2 * std::sin(2*M_PI*x);

    labels(i, 0) = y > val ? 1 : 0;
  }

  // Generate model
  bi::Rvm model = bi::Rvm(data, labels);


  // Save to csv files
  std::ofstream data_file("data.csv");
  for (int i=0; i<n_samples; i++) {
    data_file << data(i, 0) << "," << data(i, 1) << std::endl;
  }
  data_file.close();

  std::ofstream label_file("labels.csv");
  for (int i=0; i<n_samples; i++) {
    label_file << labels(i) << std::endl;
  }
  label_file.close();

  return 0;
}
