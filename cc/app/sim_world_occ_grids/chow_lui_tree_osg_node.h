#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/sim_world_occ_grids/chow_lui_tree.h"
#include "app/sim_world_occ_grids/joint_model.h"

namespace app {
namespace sim_world_occ_grids {

class ChowLuiTreeOSGNode : public osg::Group {
 public:
  ChowLuiTreeOSGNode(const ChowLuiTree &clt);

 private:
  void Render(const rt::Location &loc, const ChowLuiTree &clt);
};

} // namespace sim_world_occ_grids
} // namespace app
