#include "app/kitti_occ_grids/map_node.h"

#include "library/osg_nodes/colorful_box.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

MapNode::MapNode(const DetectionMap &dm) : osg::Group() {
  // Get map size
  int min_i = -dm.GetNX() / 2;
  int max_i = min_i + dm.GetNX();

  int min_j = -dm.GetNY() / 2;
  int max_j = min_j + dm.GetNY();

  for (int i=min_i; i<max_i; i++) {
    for (int j=min_j; j<max_j; j++) {
      double x = i * dm.GetResolution();
      double y = j * dm.GetResolution();

      double p_car = dm.GetProbability(i, j, "Car");
      double p_cyc = dm.GetProbability(i, j, "Cyclist");
      double p_ped = dm.GetProbability(i, j, "Pedestrian");
      double p_bak = dm.GetProbability(i, j, "Background");

      if (p_bak > 0.75) {
        continue;
      }

      double alpha = (1 - p_bak) * 0.8;

      osg::Vec4 color(p_car, p_cyc, p_ped, alpha);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, dm.GetResolution() * 0.9);
      addChild(box);
    }
  }
}

MapNode::MapNode(const Detector &detector) : osg::Group() {
  for (double x = -detector.GetRangeX(); x < detector.GetRangeX(); x += detector.GetRes()) {
    for (double y = -detector.GetRangeY(); y < detector.GetRangeY(); y += detector.GetRes()) {
      double p = detector.GetScore(ObjectState(x, y, 0));

      double lo = -log(1/p - 1);
      double alpha = (lo/1e2) + 0.5;
      if (alpha < 0) {
        alpha = 0;
      }

      if (alpha > 1) {
        alpha = 1;
      }

      osg::Vec4 color(0.9, 0.1, 0.1, alpha);
      osg::Vec3 pos(x, y, 0.0);

      osg::ref_ptr<osgn::ColorfulBox> box = new osgn::ColorfulBox(color, pos, 0.25); // TODO Magic Number
      addChild(box);
    }
  }
}

} // namespace kitti_occ_grids
} // namespace app
