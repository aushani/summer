// adapted from dascar
#pragma once

#include <thread>

#include <QtGui/QMainWindow>
#include <osg/MatrixTransform>
#include <osgQt/GraphicsWindowQt>
#include <osgViewer/CompositeViewer>

#include "library/viewer/viewer_widget.h"

namespace library {
namespace viewer {

class ViewerWindow : public QMainWindow {
  Q_OBJECT

 public:
  ViewerWindow(osg::ArgumentParser *args, QWidget* parent, Qt::WindowFlags f);

  int Start();

  void AddChild(osg::Node *n);

 public slots:
  void SlotCleanup();

 private:
  osg::ref_ptr<ViewerWidget> vwidget_;
  std::thread run_thread_;
  osg::ref_ptr<osg::MatrixTransform> xform_car_;

  int bin_at_ = 0;

  void Init(osg::ArgumentParser *args);

  void RunThread();
};

}  // namespace viewer
}  // namespace library
