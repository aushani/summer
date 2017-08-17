// Adapted from dascar
#include "library/osg_nodes/occ_grid.h"

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/composite_shape_group.h"

namespace library {
namespace osg_nodes {

OccGrid::OccGrid(const rt::OccGrid &og, double thresh_lo) : osg::Group() {
  // Iterate over occ grid and add occupied cells
  for (size_t i = 0; i < og.GetLocations().size(); i++) {
    rt::Location loc = og.GetLocations()[i];
    float val = og.GetLogOdds()[i];

    if (val <= thresh_lo) {
      continue;
    }

    double scale = og.GetResolution() * 0.75;

    double x = loc.i * og.GetResolution();
    double y = loc.j * og.GetResolution();
    double z = loc.k * og.GetResolution();

    double alpha = val*2;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 0.8) {
      alpha = 0.8;
    }

    osg::ref_ptr<CompositeShapeGroup> csg_box = new CompositeShapeGroup();
    csg_box->GetSDrawable()->setColor(osg::Vec4(0.1, 0.9, 0.1, alpha));

    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(x, y, z), scale);
    csg_box->GetCShape()->addChild(box);
    addChild(csg_box);
  }
}

}  // namespace osg_nodes
}  // namespace library
