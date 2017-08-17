#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const Detector &detector) : osg::Group() {

  const std::map<ObjectState, double> scores = detector.GetScores();

  for (auto it = scores.cbegin(); it != scores.cend(); it++) {
    auto os = it->first;
    double p = it->second;

    double lo = -log(1/p - 1);
    double alpha = (lo/1e2) + 0.5;
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 1) {
      alpha = 1;
    }

    osg::Vec4 color(0.9, 0.1, 0.1, alpha);
    osg::Vec3 pos(os.x, os.y, 0.0);

    osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.25); // TODO Magic Number
    addChild(box);
  }
}

} // namespace kitti_occ_grids
} // namespace app
