#pragma once

namespace library {
namespace geometry {

struct Point {
  float x;
  float y;

  Point() : x(0.0f), y(0.0f) {;}
  Point(float xx, float yy) : x(xx), y(yy) {;}
};

}
}
