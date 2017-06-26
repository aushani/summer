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
  int count_max_free = 10;

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
    gt_labels->push_back(occu ? -1.0 : 1.0);
  }
}

void make_query_points(SimWorld &sim, std::vector<hm::Point> *query_points, std::vector<float> *gt_labels) {
  for (double x = -11; x<11; x+=0.1) {
    for (double y = -11; y<11; y+=0.1) {
      hm::Point p(x, y);
      query_points->push_back(p);
      gt_labels->push_back(sim.IsOccupied(x, y) ? -1.0 : 1.0);
    }
  }
}

void am_i_clever() {

  // Create sim world and generate data
  SimWorld sim;
  std::vector<hm::Point> points;
  std::vector<float> labels;
  make_query_points(sim, &points, &labels);

  // This kernel is actually w
  LearnedKernel kernel(10.0, 0.1);
  for (size_t i = 0; i<kernel.GetDimSize(); i++) {
    for (size_t j = 0; j<kernel.GetDimSize(); j++) {
      kernel.SetPixel(i, j, -1.0f);
    }
  }

  for (const Box& box : sim.GetObjects()) {
    float x = box.GetCenterX();
    float y = box.GetCenterY();

    int i = x * kernel.GetResolution();
    int j = y * kernel.GetResolution();

    kernel.SetPixel(i, j, 1.0f);
  }

  // Make a map with this "kernel"
  hm::HilbertMap map(points, labels, kernel);
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  am_i_clever();
  return 0;

  // Create sim world and generate data
  SimWorld sim;
  std::vector<hm::Point> hits, origins;
  sim.GenerateSimData(&hits, &origins);

  // Write out data
  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  // Create query points and labels
  std::vector<hm::Point> query_points, scoring_points;
  std::vector<float> gt_labels, query_labels;
  make_scoring_points(sim, &scoring_points, &gt_labels);
  make_query_points(sim, &query_points, &query_labels);

  // Make kernel and map
  //hm::SparseKernel kernel(1.0);
  hm::BoxKernel box_kernel(1.0);
  hm::HilbertMap box_map(hits, origins, box_kernel);
  double box_score = box_map.ComputeLogLikelihood(scoring_points, gt_labels) / scoring_points.size();
  printf("Score for box kernel is %5.3f\n", box_score);

  LearnedKernel kernel(2.0, 0.5);

  LearnedKernel best_kernel(2.0, 0.5);
  best_kernel.CopyFrom(hm::BoxKernel(1.0));
  hm::HilbertMap map_bk(hits, origins, kernel);
  double best_score = map_bk.ComputeLogLikelihood(scoring_points, gt_labels) / scoring_points.size();

  int n = kernel.GetDimSize();
  uint64_t steps = 1;
  steps <<= n*n;

  for (uint64_t step=0; step<steps; step++) {

    if (step % 1000 == 0) {
      printf("Step %ld of %ld\n", step, steps);
    }

    // Make kernel
    for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
        int idx = i*n + j;
        int val = step & (1<<idx);
        kernel.SetPixel(i, j, val ? 1.0f:0.0f);
      }
    }

  //for (int i=0; i<100; i++) {
  //  hm::BoxKernel bk(1.0);
  //  kernel.CopyFrom(bk);

    // Eval kernel
    hm::HilbertMap map(hits, origins, kernel);
    double score = map.ComputeLogLikelihood(scoring_points, gt_labels) / scoring_points.size();

    if (score < best_score) {
      best_score = score;
      best_kernel.CopyFrom(kernel);
      printf("\tscore is %7.5f\n", score);
    }
  }

  hm::HilbertMap map(hits, origins, best_kernel);

  // Evaluate
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
  probs = map.GetOccupancy(scoring_points);
  for (size_t i=0; i<scoring_points.size(); i++) {
    float x = scoring_points[i].x;
    float y = scoring_points[i].y;
    float p = probs[i];

    scoring_file << x << ", " << y << ", " << p << std::endl;
  }
  scoring_file.close();

  // Write kernel to file
  std::ofstream kernel_file;
  kernel_file.open("kernel.csv");
  for (float x = -best_kernel.MaxSupport(); x<=best_kernel.MaxSupport(); x+=0.01) {
    for (float y = -best_kernel.MaxSupport(); y<=best_kernel.MaxSupport(); y+=0.01) {
      float val = best_kernel.Evaluate(x, y);
      kernel_file << x << ", " << y << ", " << val << std::endl;
    }
  }
  kernel_file.close();

  printf("Done\n");

  return 0;
}
