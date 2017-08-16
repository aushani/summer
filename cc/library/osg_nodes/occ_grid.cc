// Adapted from dascar
#include "library/osg_nodes/occ_grid.h"

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/composite_shape_group.h"

namespace library {
namespace osg_nodes {

OccGrid::OccGrid(const rt::OccGrid &og) : osg::Group() {
  osg::ref_ptr<CompositeShapeGroup> csg = new CompositeShapeGroup();
  csg->GetSDrawable()->setColor(osg::Vec4(0.1, 0.9, 0.1, 0.8));

  // Iterate over occ grid and add occupied cells
  for (size_t i = 0; i < og.locations.size(); i++) {
    rt::Location loc = og.locations[i];
    float val = og.log_odds[i];

    if (val <= 0) {
      continue;
    }

    double scale = og.resolution * 0.75;

    double x = loc.i * og.resolution;
    double y = loc.j * og.resolution;
    double z = loc.k * og.resolution;

    csg->GetCShape()->addChild(new osg::Box(osg::Vec3(x, y, z), scale));
  }

  addChild(csg);
}

}  // namespace osg_nodes
}  // namespace library
