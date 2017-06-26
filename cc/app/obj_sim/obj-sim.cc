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

double score_map(SimWorld &sim, hm::HilbertMap &map) {
  std::vector<hm::Point> query_points;
  std::vector<float> gt;
  for (double x = -11; x<11; x+=0.05) {
    for (double y = -11; y<11; y+=0.05) {
      hm::Point p(x, y);
      query_points.push_back(p);
      gt.push_back(sim.IsOccupied(x, y) ? -1.0 : 1.0);
    }
  }
  std::vector<float> probs = map.GetOccupancy(query_points);

  double score = 0;
  for (size_t i=0; i<gt.size(); i++) {
    double p = probs[i];
    double y = gt[i];

    if (y > 0) {
      score += log(p);
    } else {
      score += log(1-p);
    }
  }

  return score;
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

int main(int argc, char** argv) {
  printf("Object sim\n");

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
  std::vector<hm::Point> query_points;
  std::vector<float> gt_labels;
  make_query_points(sim, &query_points, &gt_labels);

  // Make kernel and map
  //hm::SparseKernel kernel(1.0);
  //hm::BoxKernel kernel(1.0);
  LearnedKernel kernel(4.0, 0.1);
  LearnedKernel new_kernel(4.0, 0.1);

  for (int step = 0; step<5; step++) {
    printf("Step %d\n", step);

    hm::HilbertMap map(hits, origins, kernel);

    double score = map.ComputeLogLikelihood(query_points, gt_labels) / query_points.size();
    printf("\tScore is: %7.5f\n", score);

    std::vector<double> grads;
    for (int i=0; i<kernel.GetDimSize(); i++) {
      for (int j=0; j<kernel.GetDimSize(); j++) {
        // Step kernel
        float val = kernel.GetPixel(i, j);

        float new_val = val > 0 ? 0.0:1.0;
        kernel.SetPixel(i, j, new_val);
        //printf(" Kernel %d, %d is %f\n", i, j, val);

        // Make map and evaluate
        hm::HilbertMap map_delta(hits, origins, kernel);
        double score_delta = map_delta.ComputeLogLikelihood(query_points, gt_labels) / query_points.size();

        // Reset kernel
        if (score_delta < score) {
          new_kernel.SetPixel(i, j, new_val);
        } else {
          new_kernel.SetPixel(i, j, val);
        }
      }
    }

    // Update kernel
    kernel.CopyFrom(new_kernel);
  }

  hm::HilbertMap map(hits, origins, kernel);

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

  // Write kernel to file
  std::ofstream kernel_file;
  kernel_file.open("kernel.csv");
  for (float x = -kernel.MaxSupport(); x<kernel.MaxSupport(); x+=0.1) {
    for (float y = -kernel.MaxSupport(); y<kernel.MaxSupport(); y+=0.1) {
      float val = kernel.Evaluate(x, y);
      kernel_file << x << ", " << y << ", " << val << std::endl;
    }
  }
  kernel_file.close();

  printf("Done\n");

  return 0;
}
