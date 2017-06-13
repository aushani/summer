#pragma once

#include <vector>
#include <string>

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
  HilbertMap(std::vector<Point> points, std::vector<float> occupancies);
  //HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  float get_occupancy(Point p);

 private:
  DeviceData *data_;
};
