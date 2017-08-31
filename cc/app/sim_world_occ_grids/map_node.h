#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/sim_world_occ_grids/detector.h"

namespace app {
namespace sim_world_occ_grids {

class MapNode : public osg::Group {
 public:
  MapNode(const Detector &detector);
};

} // namespace sim_world_occ_grids
} // namespace app
