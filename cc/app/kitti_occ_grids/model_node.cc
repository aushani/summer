#include "app/kitti_occ_grids/model_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

ModelNode::ModelNode(const Model &model) : osg::Group() {
  auto counts = model.GetCounts();

  double res = model.GetResolution();

  for (auto it = counts.cbegin(); it != counts.cend(); it++) {
    auto loc = it->first;
    auto counter = it->second;

    double x = loc.i * res;
    double y = loc.j * res;
    double z = loc.k * res;

    double p = counter.GetProbability(true); // Get probability that it is occupied
    if (p < 0.1) {
      continue;
    }

    double alpha = p;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 1) {
      alpha = 1;
    }

    osg::Vec4 color(0.1, 0.9, 0.1, alpha);
    osg::Vec3 pos(x, y, z);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.9 * res);
    addChild(box);
  }
}

} // namespace kitti_occ_grids
} // namespace app
