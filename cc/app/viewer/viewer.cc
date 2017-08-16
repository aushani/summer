// adapted from dascar
#include "app/viewer/viewer.h"

#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"

#include <iostream>
#include <QLabel>

namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;

namespace app {
namespace viewer {

Viewer::Viewer(osg::ArgumentParser *args) :
  qapp_(new QApplication(args->argc(), args->argv())),
  vwindow_(new ViewerWindow(args, 0, Qt::Widget)) {
}

void Viewer::AddVelodyneScan(const kt::VelodyneScan &scan) {
  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(scan);
  vwindow_->AddChild(pc);

}

void Viewer::AddOccGrid(const rt::OccGrid &og) {
  osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
  vwindow_->AddChild(ogn);
}

void Viewer::Start() {
  int rc = vwindow_->Start();

  if (rc != EXIT_SUCCESS) {
    return;
  }
  qapp_->exec();
}

}  // namespace viewer
}  // namespace app
