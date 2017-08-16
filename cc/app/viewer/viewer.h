// adapted from dascar
#pragma once

#include <QtGui/QProgressBar>
#include <QtGui/QApplication>
#include <QtGui/QSpinBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QCheckBox>
#include <osg/MatrixTransform>
#include <osgQt/GraphicsWindowQt>
#include <osgViewer/CompositeViewer>

#include <QTimer>
#include <QGridLayout>

#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/ray_tracing/occ_grid.h"

#include "app/viewer/pick_handler.h"
#include "app/viewer/viewer_widget.h"
#include "app/viewer/viewer_window.h"

namespace kt = library::kitti;
namespace rt = library::ray_tracing;

namespace app {
namespace viewer {

class Viewer {
 public:
  Viewer(osg::ArgumentParser *args);

  void AddVelodyneScan(const kt::VelodyneScan &scan);
  void AddOccGrid(const rt::OccGrid &og);
  void AddTracklets(kt::Tracklets *tracklets, int frame);

  void Start();

 private:
  QApplication *qapp_;
  ViewerWindow *vwindow_;
};

} // namespace viewer
} // namespace app
