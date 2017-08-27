#pragma once

#include <vector>

namespace library {
namespace DetectionMapping {

// 2D Map of detections
class Map {
 public:
  Map(double range_x, double range_y, double resolution);
  //OccGrid(const OccGridData &ogd);

  //float GetProbability(const Location &loc) const;
  //float GetProbability(float x, float y, float z) const;

  //float GetLogOdds(const Location &loc) const;
  //float GetLogOdds(float x, float y, float z) const;

  //const std::vector<Location>& GetLocations() const;
  //const std::vector<float>& GetLogOdds() const;
  //float GetResolution() const;

  //std::map<Location, float> MakeMap() const;

  //void Save(const char* fn) const;
  //static OccGrid Load(const char* fn);

 private:
  //const OccGridData data_;
};

}  // namespace ray_tracing
}  // namespace library
