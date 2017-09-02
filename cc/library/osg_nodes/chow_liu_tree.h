#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/chow_liu_tree/dynamic_clt.h"

namespace clt = library::chow_liu_tree;

namespace library {
namespace osg_nodes {

class ChowLiuTree : public osg::Group {
 public:
  ChowLiuTree(const clt::DynamicCLT &clt);

  void Render(const std::shared_ptr<clt::CLTNode> &node);
};

} // osg_nodes
} // library
