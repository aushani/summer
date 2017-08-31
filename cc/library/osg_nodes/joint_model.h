#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/chow_liu_tree/joint_model.h"

namespace clt = library::chow_liu_tree;

namespace library {
namespace osg_nodes {

class JointModel : public osg::Group {
 public:
  JointModel(const clt::JointModel &jm);
};

} // osg_nodes
} // library
