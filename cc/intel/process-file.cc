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

#include "hilbert_map.h"

template<typename Out>
void split(const std::string &s, char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

void load(char* fn, std::vector<Point> &hits, std::vector<Point> &origins) {
  std::ifstream file;
  file.open(fn);

  std::string line;

  while (std::getline(file, line)) {
    if (line.find("FLASER") == 0) {
      std::vector<std::string> tokens = split(line, ' ');

      double x = atof(tokens[182].c_str());
      double y = atof(tokens[183].c_str());
      double t = atof(tokens[184].c_str());
      //printf("%5.3f %5.3f %5.3f\n", x, y, t);

      Point p_origin(x, y);


      for (int i=2; i<182; i++) {
        double angle_d = -90.0 + (i-2);
        double range = atof(tokens[i].c_str());
        if (range < 80) {
          double p_x = range*cos(angle_d*M_PI/180.0 + t) + x;
          double p_y = range*sin(angle_d*M_PI/180.0 + t) + y;
          Point p_hit(p_x, p_y);
          hits.push_back(p_hit);
          origins.push_back(p_origin);
        }
      }
    }
  }

  file.close();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Need filename" << std::endl;
    return 1;
  }

  std::vector<Point> hits, origins;
  auto tic_load = std::chrono::steady_clock::now();
  load(argv[1], hits, origins);
  auto toc_load = std::chrono::steady_clock::now();
  auto t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_load - tic_load);
  printf("Loaded %ld points in %ld ms\n", hits.size(), t_load_ms.count());

  HilbertMap map(hits, origins);

  auto tic_build = std::chrono::steady_clock::now();
  int trials = 100;
  for (int i=0; i<trials; i++)
    HilbertMap tmp(hits, origins);
  auto toc_build = std::chrono::steady_clock::now();
  auto t_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_build - tic_build);
  printf("Took %5.3f ms avg for build\n", ((double)t_build_ms.count())/trials);

  std::vector<Point> query_points;
  std::vector<float> probs;

  auto tic = std::chrono::steady_clock::now();
  for (double x = -25; x<25; x+=0.1) {
    for (double y = -25; y<25; y+=0.1) {
      Point p(x, y);
      query_points.push_back(p);
      probs.push_back(map.get_occupancy(p));
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
