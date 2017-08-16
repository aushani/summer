// adapted from dascar
#include "library/viewer/viewer.h"

#include <iostream>
#include <QLabel>

namespace library {
namespace viewer {

Viewer::Viewer(osg::ArgumentParser *args) :
  qapp_(new QApplication(args->argc(), args->argv())),
  vwindow_(new ViewerWindow(args, 0, Qt::Widget)) {
}

void Viewer::AddChild(osg::Node *n) {
  vwindow_->AddChild(n);
}

void Viewer::Start() {
  int rc = vwindow_->Start();

  if (rc != EXIT_SUCCESS) {
    return;
  }
  qapp_->exec();
}

}  // namespace viewer
}  // namespace library
