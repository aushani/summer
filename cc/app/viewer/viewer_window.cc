#include "app/viewer/viewer_window.h"

#include <iostream>

#include <osgGA/NodeTrackerManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/TerrainManipulator>
#include <osgViewer/ViewerEventHandlers>

#include "library/kitti/velodyne_scan.h"
#include "library/osg_nodes/point_cloud.h"

#include "app/viewer/pick_handler.h"

namespace kt = library::kitti;
namespace osgn = library::osg_nodes;

namespace app {
namespace viewer {

ViewerWindow::ViewerWindow(osg::ArgumentParser *args, QWidget *parent, Qt::WindowFlags f) : QMainWindow(parent, f) {

  osgViewer::ViewerBase::ThreadingModel tm = osgViewer::ViewerBase::SingleThreaded;
  vwidget_ = new ViewerWidget(0, Qt::Widget, tm);

  setCentralWidget(vwidget_);

  setWindowTitle(tr("Viewer"));
  setMinimumSize(640, 480);

  Init(args);
}

ViewerWindow::~ViewerWindow() {

}

void ViewerWindow::Init(osg::ArgumentParser *args) {
  osg::ApplicationUsage *au = args->getApplicationUsage();

  osg::ref_ptr<osgViewer::View> view = vwidget_->GetView();

  // set background to black
  // TODO: magic numbers
  view->getCamera()->setClearColor(osg::Vec4d(1, 1, 1, 0));

  osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> ksm = new osgGA::KeySwitchMatrixManipulator();

  //ksm->addMatrixManipulator('1', "TerrainTrackpad", new osgGA::TerrainTrackpadManipulator());
  ksm->addMatrixManipulator( '2', "NodeTracker", new osgGA::NodeTrackerManipulator());
  ksm->addMatrixManipulator('3', "Terrain", new osgGA::TerrainManipulator());

  // set initial camera position (for all manipulators)
  // TODO: magic numbers
  ksm->setHomePosition(osg::Vec3d(0, 0, 100), osg::Vec3d(0, 0, 0), osg::Vec3d(1, 0, 0), false);

  ksm->getUsage(*au);
  view->setCameraManipulator(ksm.get());

  // add the state manipulator
  osg::ref_ptr<osgGA::StateSetManipulator> ssm =
      new osgGA::StateSetManipulator(view->getCamera()->getOrCreateStateSet());
  ssm->getUsage(*au);
  view->addEventHandler(ssm);

  // add the stats handler
  osg::ref_ptr<osgViewer::StatsHandler> sh = new osgViewer::StatsHandler();
  sh->getUsage(*au);
  view->addEventHandler(sh);

  // add the help handler
  osg::ref_ptr<osgViewer::HelpHandler> hh = new osgViewer::HelpHandler(au);
  hh->getUsage(*au);
  view->addEventHandler(hh);

  // add the screen capture handler
  osg::ref_ptr<osgViewer::ScreenCaptureHandler> sch = new osgViewer::ScreenCaptureHandler();
  sch->getUsage(*au);
  view->addEventHandler(sch);

  // add the level of detail scale selector
  osg::ref_ptr<osgViewer::LODScaleHandler> lod = new osgViewer::LODScaleHandler();
  lod->getUsage(*au);
  view->addEventHandler(lod);

  // add the pick handler
  osg::ref_ptr<PickHandler> ph = new PickHandler();
  ph->getUsage(*au);
  view->addEventHandler(ph);

  // rotate by x until z down
  // car RH coordinate frame has x forward, z down
  // osg::Matrixd H(osg::Quat(180*k_d2r, osg::Vec3d(1,0,0)));
  osg::Matrixd H(osg::Quat(0, osg::Vec3d(1, 0, 0)));
  osg::ref_ptr<osg::MatrixTransform> xform = new osg::MatrixTransform(H);

  osg::ref_ptr<osg::MatrixTransform> xform_car = new osg::MatrixTransform();
  osg::Matrixd D(osg::Quat(M_PI, osg::Vec3d(1, 0, 0)));
  D.postMultTranslate(osg::Vec3d(-1, 0, -1.2));
  xform_car->setMatrix(D);
  //xform_car->addChild(new osg::Axes());
  xform->addChild(xform_car);

  // Load velodyne scan
  std::string home_dir = getenv("HOME");
  std::string kitti_log_dir = home_dir + "/data/kittidata/extracted/";
  if (!args->read(std::string("--kitti-log-dir"), kitti_log_dir)) {
      printf("Using default KITTI log dir: %s\n", kitti_log_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args->read(std::string("--kitti-log-date"), kitti_log_date)) {
      printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args->read(std::string("--log-num"), log_num)) {
      printf("Using default KITTI log number: %d\n", log_num);
  }

  int frame_num = 0;
  if (!args->read(std::string("--frame-num"), frame_num)) {
      printf("Using default KITTI frame number: %d\n", frame_num);
  }
  char fn[1000];
  // Load Velodyne
  sprintf(fn, "%s/%s/%s_drive_%04d_sync/velodyne_points/data/%010d.bin",
          kitti_log_dir.c_str(), kitti_log_date.c_str(), kitti_log_date.c_str(), log_num, frame_num);

  kt::VelodyneScan scan(fn);
  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(scan);
  xform_car->addChild(pc);


  // set scene
  view->setSceneData(xform);
}

int ViewerWindow::Start() {
  show();

  // start threads
  std::cout << "Starting thread..." << std::endl;
  run_thread_ = std::thread(&ViewerWindow::RunThread, this);

  return EXIT_SUCCESS;
}

void ViewerWindow::SlotCleanup() {
  printf("TODO: SlotCleanup\n");
}

void ViewerWindow::RunThread() {
  while (true) {
    printf("run thread!\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

} // namespace viewer
} // namespace app
