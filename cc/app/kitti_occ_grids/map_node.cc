#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const dt::Detector &detector) : osg::Group() {
  const double min = 0;
  const double max = 1;
  const double range = max - min;

  for (double x = -detector.GetRangeX(); x < detector.GetRangeX(); x += detector.GetResolution()) {
    for (double y = -detector.GetRangeY(); y < detector.GetRangeY(); y += detector.GetResolution()) {

      dt::ObjectState os(x, y, 0);

      double p_car = detector.GetProb("Car", os);
      double p_cyclist = detector.GetProb("Cyclist", os);
      double p_pedestrian = detector.GetProb("Pedestrian", os);

      double r = (p_car-min) / range;
      double g = (p_cyclist-min) / range;
      double b = (p_pedestrian-min) / range;

      if (r < 0) r = 0;
      if (r > 1) r = 1;

      if (g < 0) g = 0;
      if (g > 1) g = 1;

      if (b < 0) b = 0;
      if (b > 1) b = 1;

      osg::Vec4 color(r, g, b, 0.5);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, detector.GetResolution()*1.0); // TODO Magic Number
      box->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
      addChild(box);
    }
  }
}

} // namespace kitti_occ_grids
} // namespace app
