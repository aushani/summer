#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/bayesian_inference/rvm.h"
#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

namespace bi = library::bayesian_inference;
namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

Eigen::VectorXd OccGridToEigen(const rt::OccGrid &og) {
  Eigen::VectorXd res(16*16);

  for (int i=0; i<16; i++) {
    for (int j=0; j<16; j++) {
      int idx_at = i*16 + j;
      int i_at = i - 8;
      int j_at = j - 8;

      float lo = og.GetLogOdds(rt::Location(i_at, j_at, 0));

      if (lo > 0) {
        res(idx_at) = 1;
      } else if (lo < 0) {
        res(idx_at) = -1;
      } else {
        res(idx_at) = 0.0;
      }
    }
  }

  return res;
}

Eigen::MatrixXd LoadDirectory(char *dir, int samples) {
  Eigen::MatrixXd res(samples, 16*16);

  int count = 0;

  fs::path p(dir);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension().string() != ".og") {
      printf("skipping extension %s\n", it->path().extension().string().c_str());
      continue;
    }

    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());
    //printf("resolution is %5.3f\n", og.GetResolution());

    // Do the accumulating
    res.row(count) = OccGridToEigen(og);
    count++;

    if (count >= samples) {
      break;
    }
  }

  return res;
}

int main(int argc, char** argv) {
  printf("Make RVM model\n");

  if (argc < 3) {
    printf("Usage: %s dir_name_pos dir_name_neg out\n", argv[0]);
    return 1;
  }

  int n_samples_per_class = 200;

  Eigen::MatrixXd samples_pos = LoadDirectory(argv[1], n_samples_per_class);
  printf("Loaded pos\n");

  Eigen::MatrixXd samples_neg = LoadDirectory(argv[2], n_samples_per_class);
  printf("Loaded neg\n");

  Eigen::MatrixXd samples(2*n_samples_per_class, 16*16);
  samples << samples_pos, samples_neg;

  Eigen::MatrixXd labels(2*n_samples_per_class, 1);
  for (int i=0; i<2*n_samples_per_class; i++) {
    labels(i, 0) = (i < n_samples_per_class) ? 1:0;
  }

  bi::GaussianKernel kernel(10.0);
  bi::Rvm model = bi::Rvm(samples, labels, &kernel);
  model.Solve(1000);

  // Save
  printf("Done! Saving to %s...\n", argv[3]);

  return 0;
}
