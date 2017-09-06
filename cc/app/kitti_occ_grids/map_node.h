#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/detector/detector.h"

namespace dt = library::detector;

namespace app {
namespace kitti_occ_grids {

class MapNode : public osg::Group {
 public:
  MapNode(const dt::Detector &detector);
};

} // namespace kitti_occ_grids
} // namespace app
