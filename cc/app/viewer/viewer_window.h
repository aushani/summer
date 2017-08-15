#pragma once

#include <thread>

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

  int Start();

 public slots:
  void SlotCleanup();

 private:
  osg::ref_ptr<ViewerWidget> vwidget_;
  std::thread run_thread_;

  void Init(osg::ApplicationUsage* au);

  void RunThread();
};

}  // namespace viewer
}  // namespace app
