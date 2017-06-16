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

#include "library/hilbert_maps/hilbert_map.h"
#include "app/obj_sim/box.h"

void generate_sim_data(std::vector<Point> *hits, std::vector<Point> *origins) {

  // Make boxes in the world
  std::vector<Box> objects;
  objects.push_back(Box(-3.0, 1.0, 1.9, 1.9));
  objects.push_back(Box(1.0, 5.0, 1.9, 1.9));
  objects.push_back(Box(3.0, 1.0, 1.9, 1.9));

  // So rays always hit something
  objects.push_back(Box(0.0, 0.0, 20.1, 20.1));

  Point origin(0.0, -2.0);
  Point hit;

  for (double angle = 0; angle < M_PI; angle += 0.01) {
    double best_distance = -1.0;
    bool valid_hit = false;

    Point b_hit;

    for ( Box &b : objects) {
      double dist = b.GetHit(origin, angle, &b_hit);
      if (dist > 0 && (!valid_hit || dist < best_distance)) {
        hit = b_hit;
        best_distance = dist;
        valid_hit = true;
      }
    }

    if (valid_hit) {
      hits->push_back(hit);
      origins->push_back(origin);
    }
  }
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  std::vector<Point> hits, origins;
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

  HilbertMap map(hits, origins);

  std::vector<Point> query_points;
  std::vector<float> probs;

  auto tic = std::chrono::steady_clock::now();
  for (double x = -11; x<11; x+=0.05) {
    for (double y = -11; y<11; y+=0.05) {
      Point p(x, y);
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
