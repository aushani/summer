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

struct Point {
  double x;
  double y;

  Point(double xx, double yy) : x(xx), y(yy) {;}
};

struct DataPoint {
  Point p;
  double val;

  DataPoint(double xx, double yy, double v) : p(xx, yy), val(v) {;}
};

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

double k_sparse(Point p1, Point p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;

  double d2 = dx*dx + dy*dy;

  if (d2 > 1.0)
    return 0;

  double r = sqrt(d2);

  double t = M_2_PI * r;

  return (2 + cos(t)) / 3 * (1 - r) + 1/M_2_PI * sin(t);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Need filename" << std::endl;
    return 1;
  }

  std::ifstream file;
  file.open(argv[1]);

  std::string line;

  std::vector<DataPoint> data;

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
          DataPoint p_hit(p_x, p_y, 1.0);
          data.push_back(p_hit);

          // Create some randomly sampled free points
          for (int i=0; i<range; i++) {
            double random_range = unif(re) * range;
            double p_x = random_range*cos(angle_d*M_PI/180.0 + t) + x;
            double p_y = random_range*sin(angle_d*M_PI/180.0 + t) + y;
            DataPoint p_free(p_x, p_y, -1.0);
            data.push_back(p_free);
          }
        }
      }
    }
  }

  file.close();

  printf("Have %ld points\n", data.size());

  std::ofstream data_file;
  data_file.open("points.csv");
  for (const DataPoint &p : data) {
    data_file << p.p.x << ", " << p.p.y << ", " << p.val << std::endl;
  }
  data_file.close();

  return 0;
}
