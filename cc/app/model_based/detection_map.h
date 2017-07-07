#pragma once

class DetectionMap {
 public:
  DetectionMap(double size, double res);

 private:
  std::map<ge::Point, double> map_;

};
