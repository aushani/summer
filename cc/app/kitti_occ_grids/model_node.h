#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/kitti_occ_grids/model.h"

namespace app {
namespace kitti_occ_grids {

class ModelNode : public osg::Group {
 public:
  ModelNode(const Model &model);
};

} // namespace kitti_occ_grids
} // namespace app
