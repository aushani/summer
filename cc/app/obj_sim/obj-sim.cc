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

double gpu_score_map(SimWorld &sim, hm::HilbertMap &map, std::vector<hm::Point> &query_points, std::vector<float> &gt_labels) {
  return map.ComputeLogLikelihood(query_points, gt_labels);
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  SimWorld sim;
  std::vector<hm::Point> hits, origins;
  sim.GenerateSimData(&hits, &origins);

  std::vector<hm::Point> query_points;
  std::vector<float> gt_labels;
  make_query_points(sim, &query_points, &gt_labels);

  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  hm::SparseKernel kernel_1(1.0);
  hm::BoxKernel kernel_2(1.0);

  hm::HilbertMap map_1(hits, origins, kernel_1);
  hm::HilbertMap map_2(hits, origins, kernel_2);

  double score_1 = gpu_score_map(sim, map_1, query_points, gt_labels);
  double score_2 = gpu_score_map(sim, map_2, query_points, gt_labels);

  printf("Sparse vs Box kernel scores: %5.3f vs %5.3f\n", score_1, score_2);

  hm::HilbertMap *map = NULL;
  if (score_1 < score_2)
    map = &map_1;
  else
    map = &map_2;

  auto tic_score = std::chrono::steady_clock::now();
  double score = gpu_score_map(sim, *map, query_points, gt_labels);
  auto toc_score = std::chrono::steady_clock::now();
  auto t_score_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_score - tic_score);
  printf("GPU Score is: %5.3f\n", score);
  printf("Evaluated score in %ld ms\n", t_score_ms.count());

  auto tic = std::chrono::steady_clock::now();
  std::vector<float> probs = map->GetOccupancy(query_points);
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Evaluated grid in %ld ms (%5.3f ms / call)\n", t_ms.count(), ((double)t_ms.count())/query_points.size());

  std::ofstream grid_file;
  grid_file.open("grid.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = probs[i];

    grid_file << x << ", " << y << ", " << p << std::endl;
  }

  grid_file.close();

  return 0;
}
