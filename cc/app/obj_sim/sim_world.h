#pragma once

#include <vector>

#include "app/obj_sim/shape.h"
#include "library/hilbert_map/hilbert_map.h"

namespace hm = library::hilbert_map;

class SimWorld {
 public:
  SimWorld();

  void GenerateSimData(std::vector<hm::Point> *hits, std::vector<hm::Point> *origins);
  void GenerateGrid(double size, std::vector<hm::Point> *points, std::vector<float> *labels);
  void GenerateSamples(size_t trials, std::vector<hm::Point> *points, std::vector<float> *labels);

  const std::vector<Shape>& GetObjects();

  bool IsOccupied(float x, float y);

  double GetMinX() const;
  double GetMaxX() const;
  double GetMinY() const;
  double GetMaxY() const;

 private:
  std::vector<Shape> objects_;
  Shape bounding_box_;
};
