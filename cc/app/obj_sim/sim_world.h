#pragma once

#include <vector>

#include "app/obj_sim/box.h"
#include "library/hilbert_map/hilbert_map.h"

namespace hm = library::hilbert_map;

class SimWorld {
 public:
  SimWorld();

  void GenerateSimData(std::vector<hm::Point> *hits, std::vector<hm::Point> *origins);

  const std::vector<Box>& GetObjects();

  bool IsOccupied(float x, float y);

 private:
  std::vector<Box> objects_;
  Box bounding_box_;
};
