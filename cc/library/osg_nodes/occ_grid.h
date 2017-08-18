// Adapted from dascar
#pragma once

#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include "library/ray_tracing/occ_grid.h"

namespace rt = library::ray_tracing;

namespace library {
namespace osg_nodes {

//class OccGridCallback : public osg::Callback {
// public:
//  OccGridCallback();
//
//  bool run(osg::Object *object, osg::Object *data) override;
//};

class OccGrid : public osg::Group {
 public:
  OccGrid(const rt::OccGrid &og, double thresh_lo=0);
};

} // osg_nodes
} // library