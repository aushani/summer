#include "app/kitti_occ_grids/detector_handler.h"

namespace app {
namespace kitti_occ_grids {

DetectorHandler::DetectorHandler(const dt::Detector &detector) :
 library::viewer::PickHandler(),
 detector_(detector) {

}

void DetectorHandler::pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;

  bool ctrl = false;
  if (ea.getModKeyMask() && osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL) {
    ctrl = true;
  }

  if (!ctrl) {
    return;
  }

  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end(); ++hitr) {

      osg::Vec3 p = hitr->getWorldIntersectPoint();

      dt::ObjectState os(p[0], p[1], 0);

      for (const auto &cn : detector_.GetClasses()) {
        printf("Class: %s, Score: %7.3f\n", cn.c_str(), detector_.GetScore(cn, os));
        printf("Class: %s, Log-odds: %7.3f\n", cn.c_str(), detector_.GetLogOdds(cn, os));
      }

      break;
    }
  }
}

} // namespace viewer
} // namespace app
