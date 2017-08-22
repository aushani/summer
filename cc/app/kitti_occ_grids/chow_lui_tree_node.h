#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/kitti_occ_grids/chow_lui_tree.h"

namespace app {
namespace kitti_occ_grids {

class ChowLuiTreeNode : public osg::Group {
 public:
  ChowLuiTreeNode(const ChowLuiTree &clt);
};

} // namespace kitti_occ_grids
} // namespace app
