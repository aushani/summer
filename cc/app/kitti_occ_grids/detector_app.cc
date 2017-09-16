#include "app/kitti_occ_grids/detector_app.h"

#include "library/osg_nodes/car.h"
#include "library/osg_nodes/colorful_box.h"

namespace app {
namespace kitti_occ_grids {

DetectorApp::DetectorApp(osg::ArgumentParser *args, bool viewer) :
 detector_(dt::Dim(0, 60, -40, 40, kRes_)),
 og_builder_(200000, kRes_, 100.0) {

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
    // Make sure it's a directory
    if (fs::is_regular_file(it->path())) {
      continue;
    }

    // Is it a class we care about?
    std::string classname = it->path().stem().string();
    printf("classname: %s\n", classname.c_str());

    //if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background")) {
    if (! (classname == "Car" || classname == "Background")) {
      continue;
    }

    // Enter directory and look for angle bins
    for (int angle_bin = 0; angle_bin < dt::Detector::kAngleBins; angle_bin++) {
      char fn[1000];
      sprintf(fn, "angle_bin_%02d/fm.fm", angle_bin);
      fs::path p_model = it->path() / fn;

      if (fs::exists(p_model)) {
        printf("Loading class %s angle bin %d\n", classname.c_str(), angle_bin);
        auto model = ft::FeatureModel::Load(p_model.string().c_str());
        detector_.AddModel(classname, angle_bin, model);
      }
    }
  }
  printf("Loaded all models\n");

  osg::ref_ptr<DetectorHandler> dh = new DetectorHandler(detector_);
  if (viewer) {
    viewer_ = std::make_shared<vw::Viewer>(args);
    viewer_->AddHandler(dh);

    viewer_->AddHandler(this);
  }

  Process();
}

void DetectorApp::Run() {
  viewer_->Start();
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

    if (ea.getKey() == 'o') {
      render_og_ = !render_og_;
      return true;
    }

    if (ea.getKey() == 'r') {
      ProcessBackground();
      return true;
    }

    if (ea.getKey() == 'f') {
      detector_.use_features = !detector_.use_features;
      printf("Use features now: %s\n", detector_.use_features ? "ON":"OFF");
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

  process_thread_ = std::make_shared<std::thread>(&DetectorApp::Process, this);
}

bool DetectorApp::SetFrame(int f) {
  frame_at_ = f;

  bool res = frame_at_ >= kFirstFrame_ && frame_at_ < kLastFrame_;
  if (frame_at_ < kFirstFrame_) {
    frame_at_ = kFirstFrame_;
  }

  if (frame_at_ >= kLastFrame_) {
    frame_at_ = kLastFrame_ - 1;
  }

  return res;
}

kt::KittiChallengeData DetectorApp::Process() {
  kt::KittiChallengeData kcd = kt::KittiChallengeData::LoadFrame(dirname_, frame_at_);

  library::timer::Timer t;
  t.Start();
  detector_.Run(kcd.GetScan().GetHits(), kcd.GetScan().GetIntensities());
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  if (viewer_) {
    osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(kcd.GetScan());
    osg::ref_ptr<osgn::ObjectLabels> ln = new osgn::ObjectLabels(kcd.GetLabels(), kcd.GetTcv());
    osg::ref_ptr<MapNode> map_node = new MapNode(detector_, kcd);

    viewer_->RemoveAllChildren();
    viewer_->AddChild(pc);
    viewer_->AddChild(ln);
    viewer_->AddChild(map_node);

    if (render_og_) {
      auto fog = og_builder_.GenerateFeatureOccGrid(kcd.GetScan().GetHits(), kcd.GetScan().GetIntensities());
      osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(fog);
      viewer_->AddChild(ogn);
    }

    osg::ref_ptr<osg::MatrixTransform> xform_car = new osg::MatrixTransform();
    osg::Matrixd D(osg::Quat(M_PI, osg::Vec3d(1, 0, 0)));
    D.postMultTranslate(osg::Vec3d(-1, 0, -1.2));
    xform_car->setMatrix(D);
    xform_car->addChild(new osgn::Car());
    viewer_->AddChild(xform_car);

    double min_thresh = 1;
    double max_thresh = 20;

    for (const auto &d : detector_.GetDetections(min_thresh)) {
      // This is ugly, but check a few times to make sure we're not on the boundary
      if (!kcd.InCameraView(d.os.x - 1.0, d.os.y + 1.0, 0.0)) {
        continue;
      }

      if (!kcd.InCameraView(d.os.x - 1.0, d.os.y - 1.0, 0.0)) {
        continue;
      }

      if (!kcd.InCameraView(d.os.x + 1.0, d.os.y + 1.0, 0.0)) {
        continue;
      }

      if (!kcd.InCameraView(d.os.x + 1.0, d.os.y - 1.0, 0.0)) {
        continue;
      }

      double r = (d.confidence - min_thresh) / (max_thresh - min_thresh);
      if (r>1) r = 1;

      viewer_->AddChild(new osgn::ColorfulBox(osg::Vec4(r, 0, 0, 1), osg::Vec3(d.os.x, d.os.y, 2.0), 1));
    }

    char fn[1000];
    sprintf(fn, "%s/%06d.txt", my_results_dir, frame_at_);
    fs::path p(fn);

    if (fs::exists(p)) {
      auto my_labels = kt::ObjectLabel::Load(fn);
      osg::ref_ptr<osgn::ObjectLabels> my_ln = new osgn::ObjectLabels(my_labels, kcd.GetTcv(), false);
      viewer_->AddChild(my_ln);
    }
  }

  return kcd;
}

const dt::Detector& DetectorApp::GetDetector() const {
  return detector_;
}

} // namespace viewer
} // namespace app
