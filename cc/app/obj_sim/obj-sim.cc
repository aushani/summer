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

int main(int argc, char** argv) {
  printf("Object sim\n");

  SimWorld sim;
  std::vector<hm::Point> hits, origins;
  sim.GenerateSimData(&hits, &origins);

  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  hm::SparseKernel kernel(1.0);
  //hm::BoxKernel kernel(1.0);
  hm::HilbertMap map(hits, origins, kernel);

  double score = score_map(sim, map);
  printf("Score is: %5.3f\n", score);

  std::vector<hm::Point> query_points;

  auto tic = std::chrono::steady_clock::now();
  for (double x = -11; x<11; x+=0.05) {
    for (double y = -11; y<11; y+=0.05) {
      hm::Point p(x, y);
      query_points.push_back(p);
    }
  }
  std::vector<float> probs = map.GetOccupancy(query_points);
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Evaluated grid in %ld ms (%5.3f ms / call)\n", t_ms.count(), ((double)t_ms.count())/query_points.size());

  std::ofstream grid_file;
  grid_file.open("grid.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = probs[i];
    //p = sim.IsOccupied(x, y) ? 1 : 0;

    grid_file << x << ", " << y << ", " << p << std::endl;
  }

  grid_file.close();

  return 0;
}
