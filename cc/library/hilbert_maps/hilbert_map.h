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
  HilbertMap(const std::vector<Point> &hits, const std::vector<Point> &origins);
  //HilbertMap(const HilbertMap &hm);
  ~HilbertMap();

  float GetOccupancy(Point p);

 private:
  DeviceData *data_;
};
