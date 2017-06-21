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
#include "app/obj_sim/box.h"

namespace hm = library::hilbert_map;

void generate_sim_data(std::vector<hm::Point> *hits, std::vector<hm::Point> *origins) {

  // Make boxes in the world
  std::vector<Box> objects;
  objects.push_back(Box(-3.0, 1.0, 1.9, 1.9));
  objects.push_back(Box(1.0, 5.0, 1.9, 1.9));
  objects.push_back(Box(3.0, 1.0, 1.9, 1.9));

  // So rays always hit something
  objects.push_back(Box(0.0, 0.0, 20.1, 20.1));

  Eigen::Vector2d origin(0.0, -2.0);
  Eigen::Vector2d hit;

  for (double angle = 0; angle < M_PI; angle += 0.01) {
    double best_distance = -1.0;
    bool valid_hit = false;

    Eigen::Vector2d b_hit;

    for ( Box &b : objects) {
      double dist = b.GetHit(origin, angle, &b_hit);
      if (dist > 0 && (!valid_hit || dist < best_distance)) {
        hit = b_hit;
        best_distance = dist;
        valid_hit = true;
      }
    }

    if (valid_hit) {
      hits->push_back(hm::Point(hit(0), hit(1)));
      origins->push_back(hm::Point(origin(0), origin(1)));
    }
  }
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  std::vector<hm::Point> hits, origins;
  auto tic_load = std::chrono::steady_clock::now();
  generate_sim_data(&hits, &origins);
  auto toc_load = std::chrono::steady_clock::now();
  auto t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_load - tic_load);
  printf("Loaded %ld points in %ld ms\n", hits.size(), t_load_ms.count());

  std::ofstream points_file;
  points_file.open("points.csv");
  for (int i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  hm::HilbertMap map(hits, origins);

  std::vector<hm::Point> query_points;
  std::vector<float> probs;

  auto tic = std::chrono::steady_clock::now();
  for (double x = -11; x<11; x+=0.05) {
    for (double y = -11; y<11; y+=0.05) {
      hm::Point p(x, y);
      query_points.push_back(p);
      probs.push_back(map.GetOccupancy(p));
    }
  }
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
