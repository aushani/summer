#pragma once

#include <QtGui/QMainWindow>

#include <osg/MatrixTransform>
#include <osgQt/GraphicsWindowQt>
#include <osgViewer/CompositeViewer>

#include "app/viewer/viewer_widget.h"

namespace app {
namespace viewer {

class ViewerWindow : public QMainWindow {
  Q_OBJECT

 public:
  ViewerWindow(osg::ArgumentParser& args, QWidget* parent, Qt::WindowFlags f);
  ~ViewerWindow();

  int start();

 public slots:
  void slot_cleanup();

 private:
  void init(osg::ApplicationUsage* au);
  osg::ref_ptr<ViewerWidget> vwidget_;
};

}  // namespace viewer
}  // namespace app
