#include "app/viewer/viewer_window.h"

#include <iostream>

#include <osgGA/NodeTrackerManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/TerrainManipulator>
#include <osgViewer/ViewerEventHandlers>

#include "app/viewer/pick_handler.h"

namespace app {
namespace viewer {

ViewerWindow::ViewerWindow(osg::ArgumentParser &args, QWidget *parent, Qt::WindowFlags f) : QMainWindow(parent, f) {

  osgViewer::ViewerBase::ThreadingModel tm = osgViewer::ViewerBase::SingleThreaded;
  vwidget_ = new ViewerWidget(0, Qt::Widget, tm);

  setCentralWidget(vwidget_);

  setWindowTitle(tr("Viewer"));
  setMinimumSize(640, 480);

  init(args.getApplicationUsage());
}

ViewerWindow::~ViewerWindow() {

}

void ViewerWindow::init(osg::ApplicationUsage* au) {
  osg::ref_ptr<osgViewer::View> view = vwidget_->get_view();

  // set background to black
  // TODO: magic numbers
  view->getCamera()->setClearColor(osg::Vec4d(1, 1, 1, 0));

  osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> ksm = new osgGA::KeySwitchMatrixManipulator();

  //ksm->addMatrixManipulator('1', "TerrainTrackpad", new osgGA::TerrainTrackpadManipulator());
  //ksm->addMatrixManipulator('3', "Terrain", new osgGA::TerrainManipulator());

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

  // set scene
  view->setSceneData(xform);
}

int ViewerWindow::start() {
  show();

  // start threads
  std::cout << "Starting fa..." << std::endl;
  //int rc = pthread_create(&_fa_thread, &_attr, fa_thread_routine, (void*)&_state);
  //if (rc) {
  //  std::cerr << "Error creating fa_thread: " << rc << std::endl;
  //  return EXIT_FAILURE;
  //}

  return EXIT_SUCCESS;
}

void ViewerWindow::slot_cleanup() {
  printf("TODO: SlotCleanup\n");
}

} // namespace viewer
} // namespace app
