#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const dt::Detector &detector) : osg::Group() {
  for (double x = -detector.GetRangeX(); x < detector.GetRangeX(); x += detector.GetResolution()) {
    for (double y = -detector.GetRangeY(); y < detector.GetRangeY(); y += detector.GetResolution()) {

      dt::ObjectState os(x, y, 0);

      double lo_car = detector.GetLogOdds("Car", os);
      double lo_cyclist = detector.GetLogOdds("Cyclist", os);
      double lo_pedestrian = detector.GetLogOdds("Pedestrian", os);

      double r = (lo_car-10) / 30;
      double g = (lo_cyclist-10) / 30;
      double b = (lo_pedestrian-10) / 30;

      if (r < 0) r = 0;
      if (r > 1) r = 1;

      if (g < 0) g = 0;
      if (g > 1) g = 1;

      if (b < 0) b = 0;
      if (b > 1) b = 1;

      osg::Vec4 color(r, g, b, 0.5);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, detector.GetResolution()*1.0); // TODO Magic Number
      addChild(box);
    }
  }
}

} // namespace kitti_occ_grids
} // namespace app
