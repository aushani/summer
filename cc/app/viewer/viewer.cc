#include "app/viewer/viewer.h"

#include <iostream>
#include <QLabel>

namespace app {
namespace viewer {

Viewer::Viewer(osg::ArgumentParser& args) :
  qapp_(new QApplication(args.argc(), args.argv())),
  vwindow_(new ViewerWindow(args, 0, Qt::Widget)) {
}

void Viewer::Start() {
  int rc = vwindow_->start();

  if (rc != EXIT_SUCCESS) {
    return;
  }
  qapp_->exec();
}

}  // namespace viewer
}  // namespace app
