#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const dt::Detector &detector) : osg::Group() {
  for (double x = -detector.GetRangeX(); x < detector.GetRangeX(); x += detector.GetResolution()) {
    for (double y = -detector.GetRangeY(); y < detector.GetRangeY(); y += detector.GetResolution()) {

      dt::ObjectState os(x, y, 0);

      double p_car = detector.GetProb("Car", os);
      double p_cyclist = detector.GetProb("Cyclist", os);
      double p_pedestrian = detector.GetProb("Pedestrian", os);
      double p_background = detector.GetProb("Background", os);

      //if (p_background > 0.5) {
      //  continue;
      //}

      osg::Vec4 color(p_car, p_cyclist, p_pedestrian, 1-p_background);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, detector.GetResolution()*0.9); // TODO Magic Number
      addChild(box);
    }
  }
}

} // namespace kitti_occ_grids
} // namespace app
