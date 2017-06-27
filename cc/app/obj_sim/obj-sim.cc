#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <math.h>
#include <random>
#include <chrono>

#include "library/hilbert_map/hilbert_map.h"

#include "app/obj_sim/sim_world.h"
#include "app/obj_sim/learned_kernel.h"

namespace hm = library::hilbert_map;

void make_scoring_points(SimWorld &sim, std::vector<hm::Point> *query_points, std::vector<float> *gt_labels) {
  int count_occu = 0;
  int count_free = 0;
  int count_max_occu = 10000;
  int count_max_free = 10000;

  double lower_bound = -10;
  double upper_bound = 10;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  while (count_occu < count_max_occu || count_free < count_max_free) {
    double x = unif(re);
    double y = unif(re);

    bool occu = sim.IsOccupied(x, y);
    if (occu) {
      if (count_occu == count_max_occu) {
        continue;
      } else {
        count_occu++;
      }
    }
    if (!occu) {
      if (count_free == count_max_free) {
        continue;
      } else {
        count_free++;
      }
    }

    hm::Point p(x, y);
    query_points->push_back(p);
    gt_labels->push_back(occu ? 1.0 : -1.0);
  }
}

void make_query_points(SimWorld &sim, std::vector<hm::Point> *query_points, std::vector<float> *gt_labels) {
  for (double x = -11; x<11; x+=0.5) {
    for (double y = -11; y<11; y+=0.5) {
      hm::Point p(x, y);
      query_points->push_back(p);
      gt_labels->push_back(sim.IsOccupied(x, y) ? -1.0 : 1.0);
    }
  }
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  // Create sim world and generate data
  printf("Generating sim...\n");
  SimWorld sim;
  std::vector<hm::Point> points;
  std::vector<float> labels;
  //make_query_points(sim, &points, &labels);
  make_scoring_points(sim, &points, &labels);
  printf("Have %ld points to train on\n", points.size());

  // This kernel is actually w
  printf("Setting up kernel...\n");
  LearnedKernel w_kernel(20.0, 0.5);
  for (size_t i = 0; i<w_kernel.GetDimSize(); i++) {
    for (size_t j = 0; j<w_kernel.GetDimSize(); j++) {
      w_kernel.SetPixel(i, j, -1.0f);
    }
  }

  for (const Box& box : sim.GetObjects()) {
    float x = box.GetCenterX();
    float y = box.GetCenterY();

    w_kernel.SetLocation(x, y, 1.0f);
  }
  printf("Have %ldx%ld kernel\n", w_kernel.GetDimSize(), w_kernel.GetDimSize());

  // Make a map with this "kernel"
  printf("Learning map...\n");
  hm::Opt opt;
  opt.min = -2.0;
  opt.max = 2.0;
  opt.inducing_points_n_dim = 20;
  opt.learning_rate = 0.01;
  hm::HilbertMap w_map(points, labels, w_kernel, opt);

  // Now actually make a HM with the kernel we learned
  std::vector<float> w = w_map.GetW();
  LearnedKernel kernel(opt.max - opt.min, (opt.max - opt.min)/opt.inducing_points_n_dim);
  for (int i=0; i<opt.inducing_points_n_dim; i++) {
    for (int j=0; j<opt.inducing_points_n_dim; j++) {
      int idx = i * opt.inducing_points_n_dim + j;
      float val = w[idx];
      kernel.SetPixel(i, j, val);
    }
  }

  std::vector<hm::Point> hits, origins;
  sim.GenerateSimData(&hits, &origins);

  printf("Making actual map\n");
  hm::HilbertMap map(hits, origins, kernel);

  printf("Done, writing files...\n");

  // Write out data
  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  // Evaluate
  std::vector<hm::Point> query_points;
  std::vector<float> gt_labels;
  make_query_points(sim, &query_points, &gt_labels);

  auto tic = std::chrono::steady_clock::now();
  std::vector<float> probs = map.GetOccupancy(query_points);
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Evaluated grid in %ld ms (%5.3f ms / call)\n", t_ms.count(), ((double)t_ms.count())/query_points.size());

  // Write grid to file
  std::ofstream grid_file;
  grid_file.open("grid.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = probs[i];

    grid_file << x << ", " << y << ", " << p << std::endl;
  }
  grid_file.close();

  std::ofstream scoring_file;
  scoring_file.open("scoring.csv");
  for (size_t i=0; i<points.size(); i++) {
    float x = points[i].x;
    float y = points[i].y;
    float p = labels[i];

    scoring_file << x << ", " << y << ", " << p << std::endl;
  }
  scoring_file.close();

  // Write kernel to file
  std::ofstream kernel_file;
  kernel_file.open("kernel.csv");
  for (float x = -kernel.MaxSupport(); x<=kernel.MaxSupport(); x+=0.05) {
    for (float y = -kernel.MaxSupport(); y<=kernel.MaxSupport(); y+=0.05) {
      float val = kernel.Evaluate(x, y);
      kernel_file << x << ", " << y << ", " << val << std::endl;
    }
  }
  kernel_file.close();

  printf("Done\n");

  return 0;
}
