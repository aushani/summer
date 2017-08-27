#pragma once

#include <map>

#include "library/ray_tracing/occ_grid.h"

#include "app/kitti_occ_grids/detection_map.h"
#include "app/kitti_occ_grids/marginal_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class MarginalDetector {
 public:
  MarginalDetector(double resolution);

  void AddModel(const std::string &classname, const MarginalModel &mm);

  DetectionMap RunDetector(const rt::OccGrid &og) const;

 private:
  double resolution_;
  std::map<std::string, MarginalModel> models_;
  std::vector<std::string> classes_;

};

} // namespace kitti_occ_grids
} // namespace app
