#pragma once

#include <vector>
#include <string>

#include "library/hilbert_map/kernel.h"

namespace library {
namespace hilbert_map {

struct Point {
  float x;
  float y;

  Point() : x(0.0f), y(0.0f) {;}
  Point(float xx, float yy) : x(xx), y(yy) {;}
};

struct Opt {

  // Params
  float learning_rate = 0.1;
  float min = -25.0;
  float max =  25.0;

  int inducing_points_n_dim = 1000;
};

// Forward declaration
typedef struct DeviceData DeviceData;

class HilbertMap {
 public:
  HilbertMap(const std::vector<Point> &hits, const std::vector<Point> &origins, const IKernel &kernel, const Opt opt = Opt());
  HilbertMap(const std::vector<Point> &Points, const std::vector<float> &labels, const IKernel &kernel, const Opt opt = Opt());
  //HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  std::vector<float> GetOccupancy(std::vector<Point> p);
  float ComputeLogLikelihood(std::vector<Point> points, std::vector<float> gt_labels);

  std::vector<float> GetW();

 private:
  HilbertMap(DeviceData *data);
  DeviceData *data_;
};

}
}
