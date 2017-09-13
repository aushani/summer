#include "app/kitti_occ_grids/detector_app.h"

namespace app {
namespace kitti_occ_grids {

DetectorApp::DetectorApp(osg::ArgumentParser *args) :
 detector_(0.5, 50, 50),
 viewer_(args),
 og_builder_(200000, 0.5, 100.0) {

  // Get parameters
  args->read(std::string("--num"), frame_at_);

  dirname_ = "/home/aushani/data/kitti_challenge/";
  if (!args->read(std::string("--kitti-challenge-dir"), dirname_)) {
    printf("Using default KITTI dir: %s\n", dirname_.c_str());
  }

  std::string model_dir;
  bool res = args->read("--models", model_dir);
  BOOST_ASSERT(res);

  // Load models
  printf("loading models from %s\n", model_dir.c_str());

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(model_dir); it != end_it; it++) {
    // Make sure it's not a directory
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    // Make sure it's a joint model
    if (fs::extension(it->path()) != ".jm") {
      continue;
    }

    std::string classname = it->path().stem().string();

    if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background")) {
      continue;
    }

    printf("Found %s\n", classname.c_str());

    clt::JointModel jm = clt::JointModel::Load(it->path().string().c_str());
    detector_.AddModel(classname, jm);
  }
  printf("Loaded all models\n");

  osg::ref_ptr<DetectorHandler> dh = new DetectorHandler(detector_);
  viewer_.AddHandler(dh);

  viewer_.AddHandler(this);

  Process();
}

void DetectorApp::Run() {
  viewer_.Start();
}

bool DetectorApp::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) {

  if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
    if (ea.getKey() == 'n') {
      frame_at_++ ;
      if (frame_at_ >= kLastFrame_) {
        frame_at_ = kLastFrame_ - 1;
      }

      printf("Processing %d\n", frame_at_);

      ProcessBackground();
      return true;
    }

    if (ea.getKey() == 'p') {
      frame_at_-- ;
      if (frame_at_ < kFirstFrame_) {
        frame_at_ = kFirstFrame_;
      }

      printf("Processing %d\n", frame_at_);

      ProcessBackground();
      return true;
    }
  }

  return false;
}

void DetectorApp::ProcessBackground() {
  // Finish old thread, if applicable
  if (process_thread_) {
    process_thread_->join();
  }

  process_thread_ = std::make_unique<std::thread>(&DetectorApp::Process, this);
}

void DetectorApp::Process() {
  kt::KittiChallengeData kcd = kt::KittiChallengeData::LoadFrame(dirname_, frame_at_);

  library::timer::Timer t;
  t.Start();
  detector_.Run(kcd.GetScan().GetHits());
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(kcd.GetScan());
  osg::ref_ptr<osgn::ObjectLabels> ln = new osgn::ObjectLabels(kcd.GetLabels(), kcd.GetTcv());
  osg::ref_ptr<MapNode> map_node = new MapNode(detector_, kcd);

  viewer_.RemoveAllChildren();
  viewer_.AddChild(pc);
  //viewer_.AddChild(ln);
  viewer_.AddChild(map_node);
}

} // namespace viewer
} // namespace app
