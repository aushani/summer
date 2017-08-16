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

#include "app/viewer/pick_handler.h"
#include "app/viewer/viewer_widget.h"
#include "app/viewer/viewer_window.h"

namespace app {
namespace viewer {

class Viewer {
 public:
  Viewer(osg::ArgumentParser *args);

  void Start();

 private:
  QApplication *qapp_;
  ViewerWindow *vwindow_;
};

} // namespace viewer
} // namespace app
