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

void load(char* fn, std::vector<Point> &points, std::vector<double> &labels) {
  std::ifstream file;
  file.open(fn);

  std::string line;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::default_random_engine re;

  while (std::getline(file, line)) {
    if (line.find("FLASER") == 0) {
      std::vector<std::string> tokens = split(line, ' ');

      double x = atof(tokens[182].c_str());
      double y = atof(tokens[183].c_str());
      double t = atof(tokens[184].c_str());
      //printf("%5.3f %5.3f %5.3f\n", x, y, t);

      for (int i=2; i<182; i++) {
        double angle_d = -90.0 + (i-2);
        double range = atof(tokens[i].c_str());
        if (range < 80) {
          double p_x = range*cos(angle_d*M_PI/180.0 + t) + x;
          double p_y = range*sin(angle_d*M_PI/180.0 + t) + y;
          Point p_hit(p_x, p_y);
          points.push_back(p_hit);
          labels.push_back(1.0);

          // Create some randomly sampled free points
          for (int i=0; i<range; i++) {
            double random_range = unif(re) * range;
            double p_x = random_range*cos(angle_d*M_PI/180.0 + t) + x;
            double p_y = random_range*sin(angle_d*M_PI/180.0 + t) + y;
            Point p_free(p_x, p_y);
            points.push_back(p_free);
            labels.push_back(-1.0);
          }
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

  std::vector<Point> points;
  std::vector<double> labels;
  load(argv[1], points, labels);
  printf("Have %ld points\n", points.size());

  std::ofstream data_file;
  data_file.open("points.csv");
  for (size_t i=0; i<points.size(); i++) {
    Point p = points[i];
    data_file << p.x << ", " << p.y << ", " << labels[i] << std::endl;
  }
  data_file.close();

  HilbertMap map(points, labels);

  std::ofstream grid_file;
  grid_file.open("grid.csv");
  for (double x = -23; x<23; x+=0.1) {
    for (double y = -23; y<23; y+=0.1) {
      double p = map.get_occupancy(Point(x, y));
      grid_file << x << ", " << y << ", " << p << std::endl;
    }
  }

  grid_file.close();

  return 0;
}
