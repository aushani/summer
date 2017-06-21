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

// Forward declaration
typedef struct DeviceData DeviceData;

class HilbertMap {
 public:
  HilbertMap(const std::vector<Point> &hits, const std::vector<Point> &origins, const IKernel &kernel);
  //HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  std::vector<float> GetOccupancy(std::vector<Point> p);

 private:
  DeviceData *data_;
};

}
}
