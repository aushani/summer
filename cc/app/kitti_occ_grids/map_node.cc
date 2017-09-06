#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const dt::Detector &detector) : osg::Group() {
  for (double x = -detector.GetRangeX(); x < detector.GetRangeX(); x += detector.GetRes()) {
    for (double y = -detector.GetRangeY(); y < detector.GetRangeY(); y += detector.GetRes()) {

      double p_car = detector.GetProb("Car", dt::ObjectState(x, y, 0));
      double p_cyclist = detector.GetProb("Cyclist", dt::ObjectState(x, y, 0));
      double p_pedestrian = detector.GetProb("Pedestrian", dt::ObjectState(x, y, 0));
      double p_background = 1 - p_car - p_cyclist - p_background;

      if (p_background > 0.9) {
        continue;
      }

      osg::Vec4 color(p_car, p_cyclist, p_pedestrian, 1-p_background);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.25); // TODO Magic Number
      addChild(box);
    }
  }
}

} // namespace kitti_occ_grids
} // namespace app
