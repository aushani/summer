#pragma once

#include "detection_map.h"

class Detector {
 public:
  Detector();

  DetectionMap Detect(const std::vector<ge::Point> &points, const std::vector<float> &labels);

};
