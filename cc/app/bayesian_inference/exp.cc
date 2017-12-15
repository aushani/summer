#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>

#include <Eigen/Core>

#include "library/timer/timer.h"

#include "library/bayesian_inference/rvm.h"
#include "library/bayesian_inference/kernel.h"

namespace bi = library::bayesian_inference;
namespace tr = library::timer;

int main(int argc, char **argv) {
  printf("RVM Test\n");
  printf("./bin [training samples] [test samples] [dimension] [iterations]\n");

  int n_train_samples = 250;
  int n_test_samples = 1000;
  int dim = 2;
  int iterations = 100;

  if (argc > 1) {
    n_train_samples = atoi(argv[1]);
  }

  if (argc > 2) {
    n_test_samples = atoi(argv[2]);
  }

  if (argc > 3) {
    dim = atoi(argv[3]);
  }

  if (argc > 4) {
    iterations = atoi(argv[4]);
  }

  tr::Timer t;

  printf("%d training samples, %d test samples, dimensionality %d, %d iterations\n",
      n_train_samples, n_test_samples, dim, iterations);

  std::vector<Eigen::VectorXd> train_data;
  std::vector<int> train_labels;

  std::vector<Eigen::VectorXd> test_data;
  std::vector<int> test_labels;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  // Generate samples
  //std::normal_distribution<double> distribution(0, 0.3);
  std::uniform_real_distribution<> distribution(-0.75, 0.75);

  for (int i=0; i<n_train_samples; i++) {
    Eigen::VectorXd sample(dim);

    for (int j=0; j<dim; j++) {
      sample(j) = distribution(rand_engine);
    }
    train_data.push_back(sample);

    double x = sample(0);
    double y = sample(1);
    double val = 0.2 * std::sin(2*M_PI*x);

    train_labels.push_back(y > val ? 1 : 0);
  }

  for (int i=0; i<n_test_samples; i++) {
    Eigen::VectorXd sample(dim);

    for (int j=0; j<dim; j++) {
      sample(j) = distribution(rand_engine);
    }
    test_data.push_back(sample);

    double x = sample(0);
    double y = sample(1);
    double val = 0.2 * std::sin(2*M_PI*x);

    test_labels.push_back(y > val ? 1 : 0);
  }

  // Generate model
  bi::GaussianKernel kernel(0.5);
  bi::Rvm<Eigen::VectorXd> model = bi::Rvm<Eigen::VectorXd>(train_data, train_labels, &kernel);

  t.Start();
  model.Solve(iterations);
  printf("Took %5.3f sec to train RVM\n", t.GetSeconds());

  t.Start();
  std::vector<double> pred_labels = model.PredictLabels(test_data);
  printf("Took %5.3f ms to predict %d labels\n", t.GetMs(), n_test_samples);

  // Save to csv files
  std::ofstream data_file("data.csv");
  for (const auto &test_sample : test_data) {
    data_file << test_sample(0) << "," << test_sample(1) << std::endl;
  }
  data_file.close();

  std::ofstream label_file("labels.csv");
  for (int i=0; i<n_test_samples; i++) {
    label_file << test_labels[i] << "," << pred_labels[i] << std::endl;
  }
  label_file.close();

  std::ofstream xm_file("xm.csv");
  std::vector<Eigen::VectorXd> x_ms = model.GetRelevanceVectors();
  for (const auto &x_m : x_ms) {
    xm_file << x_m(0) << "," << x_m(1) << std::endl;
  }
  xm_file.close();

  return 0;
}
