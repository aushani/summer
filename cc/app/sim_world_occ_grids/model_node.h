#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "app/sim_world_occ_grids/model.h"
#include "app/sim_world_occ_grids/joint_model.h"

namespace app {
namespace sim_world_occ_grids {

class ModelNode : public osg::Group {
 public:
  ModelNode(const JointModel &jm);
  ModelNode(const Model &model);
};

} // namespace sim_world_occ_grids
} // namespace app
