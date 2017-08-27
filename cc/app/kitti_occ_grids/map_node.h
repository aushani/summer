#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/kitti_occ_grids/detector.h"
#include "app/kitti_occ_grids/detection_map.h"

namespace app {
namespace kitti_occ_grids {

class MapNode : public osg::Group {
 public:
  MapNode(const DetectionMap &dm);
  MapNode(const Detector &detector);
};

} // namespace kitti_occ_grids
} // namespace app
