#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/composite_shape_group.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const Detector &detector) : osg::Group() {

  const std::map<ObjectState, double> scores = detector.GetScores();

  // Find range
  double min_score = 0;
  double max_score = 0;
  bool first = true;

  for (auto it = scores.cbegin(); it != scores.cend(); it++) {
    double score = it->second;

    if (first || score < min_score) {
      min_score = score;
    }

    if (first || score > max_score) {
      max_score = score;
    }
  }

  for (auto it = scores.cbegin(); it != scores.cend(); it++) {
    auto os = it->first;
    double score = it->second;

    double alpha = (score - min_score) / (max_score - min_score);
    if (alpha < 0) {
      alpha = 0;
    }

    if (alpha > 0.8) {
      alpha = 0.8;
    }

    osg::ref_ptr<osgn::CompositeShapeGroup> csg_box = new osgn::CompositeShapeGroup();
    csg_box->GetSDrawable()->setColor(osg::Vec4(0.9, 0.1, 0.1, alpha));

    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(os.x, os.y, 10.0), 0.25); // TODO Magic number
    csg_box->GetCShape()->addChild(box);
    addChild(csg_box);
  }
}

} // namespace kitti_occ_grids
} // namespace app
