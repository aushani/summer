#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/chow_liu_tree/joint_model.h"
#include "library/chow_liu_tree/marginal_model.h"

namespace clt = library::chow_liu_tree;

namespace library {
namespace osg_nodes {

class OccModel : public osg::Group {
 public:
  OccModel(const clt::JointModel &jm);
  OccModel(const clt::MarginalModel &mm);
};

} // osg_nodes
} // library
