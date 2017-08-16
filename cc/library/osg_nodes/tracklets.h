// Adapted from dascar
#pragma once

#include <osg/MatrixTransform>

#include "library/kitti/tracklets.h"

namespace library {
namespace osg_nodes {

class Tracklets : public osg::Group {
 public:
  Tracklets(library::kitti::Tracklets *tracklets, int frame);
};

}  // namespace osg_nodes
}  // namespace library
