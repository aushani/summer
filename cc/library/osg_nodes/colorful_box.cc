// Adapted from dascar
#include "library/osg_nodes/colorful_box.h"

#include <osg/BlendFunc>

namespace library {
namespace osg_nodes {

ColorfulBox::ColorfulBox(osg::Vec4 color) : osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0), 0.5)) {
  setColor(color);

  osg::StateSet* set = getOrCreateStateSet();
  set->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  set->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA), osg::StateAttribute::ON);
}

}  // namespace osg_nodes
}  // namespace library
