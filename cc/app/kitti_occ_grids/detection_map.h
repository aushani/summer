#pragma once

#include <unordered_map>
#include <string>
#include <vector>

namespace app {
namespace kitti_occ_grids {

class DetectionMap {
 public:
  DetectionMap(double range_x, double range_y, double resolution, const std::vector<std::string> &classes);

  void Update(int i, int j, const std::string &classname, double update);

  double GetScore(int i, int j, const std::string &classname) const;
  double GetProbability(int i, int j, const std::string &classname) const;

  std::vector<std::string> GetClasses() const;
  size_t GetNX() const;
  size_t GetNY() const;
  double GetResolution() const;

 private:
  double resolution_;

  std::unordered_map<std::string, int> classes_;

  size_t n_classes_;
  size_t n_x_;
  size_t n_y_;

  std::vector<double> scores_;

  int GetIndex(int i, int j, const std::string &classname) const;

};

} // namespace kitti_occ_grids
} // namespace app
