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
#include "library/hilbert_map/kernel.h"

namespace hm = library::hilbert_map;
namespace ge = library::geometry;

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

void load(char* fn, std::vector<ge::Point> &hits, std::vector<ge::Point> &origins) {
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

      ge::Point p_origin(x, y);


      for (int i=2; i<182; i++) {
        double angle_d = -90.0 + (i-2);
        double range = atof(tokens[i].c_str());
        if (range < 80) {
          double p_x = range*cos(angle_d*M_PI/180.0 + t) + x;
          double p_y = range*sin(angle_d*M_PI/180.0 + t) + y;
          ge::Point p_hit(p_x, p_y);
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

  std::vector<ge::Point> hits, origins;
  auto tic_load = std::chrono::steady_clock::now();
  load(argv[1], hits, origins);
  auto toc_load = std::chrono::steady_clock::now();
  auto t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_load - tic_load);
  printf("Loaded %ld points in %ld ms\n", hits.size(), t_load_ms.count());

  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  hm::SparseKernel kernel(0.5);
  std::vector<hm::IKernel*> kernels;
  kernels.push_back(&kernel);
  hm::HilbertMap map(hits, origins, kernels);

  auto tic_build = std::chrono::steady_clock::now();
  int trials = 10;
  for (int i=0; i<trials; i++)
    hm::HilbertMap tmp(hits, origins, kernels);
  auto toc_build = std::chrono::steady_clock::now();
  auto t_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc_build - tic_build);
  printf("Took %5.3f ms avg for build\n", ((double)t_build_ms.count())/trials);

  std::vector<ge::Point> query_points;

  auto tic = std::chrono::steady_clock::now();
  for (double x = -25; x<25; x+=0.1) {
    for (double y = -25; y<25; y+=0.1) {
      ge::Point p(x, y);
      query_points.push_back(p);
    }
  }
  std::vector<float> probs = map.GetOccupancy(query_points);
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Evaluated grid in %ld ms (%5.3f ms / call)\n", t_ms.count(), ((double)t_ms.count())/query_points.size());

  std::ofstream hm_file;
  hm_file.open("hilbert_map.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = probs[i];

    hm_file << x << ", " << y << ", " << p << std::endl;
  }

  hm_file.close();

  return 0;
}
