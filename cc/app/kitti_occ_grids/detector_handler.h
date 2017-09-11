#pragma once

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osgViewer/CompositeViewer>

#include "library/detector/detector.h"
#include "library/viewer/pick_handler.h"

namespace dt = library::detector;

namespace app {
namespace kitti_occ_grids {

// from osgpick example
// class to handle events with a pick
class DetectorHandler : public library::viewer::PickHandler {
 public:
  DetectorHandler(const dt::Detector &detector);

  void pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);

 private:
  const dt::Detector &detector_;
};

} // namespace viewer
} // namespace app
