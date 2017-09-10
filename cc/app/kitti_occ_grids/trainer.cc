#include "app/kitti_occ_grids/trainer.h"

#include <boost/format.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/osg_nodes/object_labels.h"
#include "library/timer/timer.h"

#include "app/kitti_occ_grids/map_node.h"

namespace osgn = library::osg_nodes;

namespace app {
namespace kitti_occ_grids {

Trainer::Trainer(const std::string &save_base_fn) :
 save_base_path_(save_base_fn),
 detector_(kRes_, 50, 50),
 og_builder_(200000, kRes_, 100.0) {

  // Configure occ grid builder size
  og_builder_.ConfigureSizeInPixels(7, 7, 5);

  models_.insert({"Car", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Cyclist", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Pedestrian", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Background", clt::JointModel(3.0, 2.0, kRes_)});

  for (const auto &kv : models_) {
    detector_.AddModel(kv.first, kv.second);
  }

  printf("Initialized all models\n");

  for (double x = -detector_.GetRangeX(); x < detector_.GetRangeX(); x += detector_.GetResolution()) {
    for (double y = -detector_.GetRangeY(); y < detector_.GetRangeY(); y += detector_.GetResolution()) {
      dt::ObjectState os(x, y, 0);
      states_.emplace_back(x, y, 0);
    }
  }
}

void Trainer::LoadFrom(const std::string &load_base_dir) {
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(load_base_dir); it != end_it; it++) {
    // Make sure it's not a directory
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    // Make sure it's a joint model
    if (fs::extension(it->path()) != ".jm") {
      continue;
    }

    std::string classname = it->path().stem().string();

    if (models_.count(classname) == 0) {
      continue;
    }

    printf("Found %s\n", classname.c_str());
    clt::JointModel jm = clt::JointModel::Load(it->path().string().c_str());

    detector_.UpdateModel(classname, jm);
  }
  printf("Loaded all models\n");
}

void Trainer::SetViewer(const std::shared_ptr<vw::Viewer> &viewer) {
  viewer_ = viewer;
}

void Trainer::Run(int first_epoch, int first_frame) {
  int epoch = first_epoch;
  int starting_frame = first_frame;

  while (true) {
    library::timer::Timer t;
    for (int frame = starting_frame; frame <= kNumFrames; frame++) {
      t.Start();
      ProcessFrame(frame);
      printf("  Processed frame %04d in %9.3f sec\n", frame, t.GetSeconds());

      // Save?
      if (frame % 100 == 0) {
        t.Start();

        fs::path dir = save_base_path_ / (boost::format("%|04|_%|06|") % epoch % frame).str();
        fs::create_directories(dir);
        for (auto &kv : models_) {
          fs::path fn = dir / (kv.first + ".jm");

          // Load back from detector
          detector_.LoadIntoJointModel(kv.first, &kv.second);

          // Now save
          kv.second.Save(fn.string().c_str());
          printf("\tSaved model to %s\n", fn.string().c_str());
        }

        printf("Took %9.3f ms to save models\n", t.GetMs());
      }
    }

    epoch++;
    starting_frame = 0;
  }
}

void Trainer::RunBackground(int first_epoch, int first_frame_num) {
  run_thread_ = std::thread(&Trainer::Run, this, first_epoch, first_frame_num);
}

std::string Trainer::GetTrueClass(const kt::KittiChallengeData &kcd, const dt::ObjectState &os) const {
  Eigen::Vector3d x_vel(os.x, os.y, 0);
  Eigen::Vector4d x_camera = kcd.GetTcv() * x_vel.homogeneous();

  for (const auto &label : kcd.GetLabels()) {
    if (!label.Care()) {
      continue;
    }

    Eigen::Vector3d x_object = (label.H_camera_object * x_camera).hnormalized();

    // Check if we're inside this object, otherwise this is not the object we
    // are looking for...
    double width = label.dimensions[1];
    double length = label.dimensions[2];

    // Add a resolution's worth of buffer
    double dl = std::fabs(x_object.x()) - kRes_; // Yes, z
    double dw = std::fabs(x_object.z()) - kRes_;
    if (dl<length/2 && dw<width/2) {
      // TODO inside box or just within kRes_ of center????
      return kt::ObjectLabel::GetString(label.type);
    }
  }

  // This is background
  return "Background";
}

std::vector<Trainer::Sample> Trainer::GetTrainingSamples(const kt::KittiChallengeData &kcd) const {
  std::vector<Sample> samples;
  double total_weight = 0;

  for (const auto &os : states_) {
    // This is ugly, but check a few times to make sure we're not on the boundary
    if (!kcd.InCameraView(os.x - 1.0, os.y + 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(os.x - 1.0, os.y - 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(os.x + 1.0, os.y + 1.0, 0.0)) {
      continue;
    }

    if (!kcd.InCameraView(os.x + 1.0, os.y - 1.0, 0.0)) {
      continue;
    }

    std::string classname = GetTrueClass(kcd, os);

    // Is this one of the classes we care about?
    // If not, ignore for now
    if (models_.count(classname) == 0) {
      //printf("No model for %s\n", classname.c_str());
      continue;
    }

    // Check score, if score = 0 no evidence, not worth pursing
    if (detector_.GetScore(classname, os) == 0) {
      continue;
    }

    double p_class = detector_.GetProb(classname, os);

    Sample s(p_class, os, classname);
    samples.push_back(s);
    total_weight += s.p_wrong;
  }

  std::random_shuffle(samples.begin(), samples.end());
  std::vector<Sample> chosen_samples;

  double weight_rollover = total_weight / 100;
  double weight_at = 0.0;
  for (const auto &s : samples) {
    weight_at += s.p_wrong;
    if (weight_at > weight_rollover) {
      weight_at -= weight_rollover;
      chosen_samples.push_back(s);
    }
  }

  return chosen_samples;
}

void Trainer::ProcessFrame(int frame) {
  library::timer::Timer t;

  // Get data
  kt::KittiChallengeData kcd = kt::KittiChallengeData::LoadFrame(kKittiBaseFilename, frame);

  // Run detector
  t.Start();
  detector_.Run(kcd.GetScan().GetHits());
  printf("\tTook %9.3f ms to run detector\n", t.GetMs());

  // Get training samples, find out where it's more wrong
  t.Start();
  std::vector<Sample> samples = GetTrainingSamples(kcd);
  printf("\tTook %9.3f ms to get %ld training samples\n", t.GetMs(), samples.size());

  // Update joint models in detector
  t.Start();
  for (const Sample &s : samples) {
    // Make occ grid
    og_builder_.SetPose(Eigen::Vector3d(s.os.x, s.os.y, 0), 0); // TODO rotation
    auto dog = og_builder_.GenerateOccGridDevice(kcd.GetScan().GetHits());

    detector_.UpdateModel(s.classname, *dog);
    samples_per_class_[s.classname]++;

    // Cleanup
    dog->Cleanup();
  }
  printf("\tTook %9.3f ms to update joint models\n", t.GetMs());

  // If we have a viewer, update render now
  if (viewer_) {
    t.Start();
    osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(kcd.GetScan());
    osg::ref_ptr<osgn::ObjectLabels> labels = new osgn::ObjectLabels(kcd.GetLabels(), kcd.GetTcv());
    osg::ref_ptr<MapNode> map_node = new MapNode(detector_);

    viewer_->RemoveAllChildren();
    viewer_->AddChild(pc);
    viewer_->AddChild(labels);
    viewer_->AddChild(map_node);

    // Add samples
    for (const Sample &s : samples) {
      osg::ref_ptr<osgn::ColorfulBox> box
        = new osgn::ColorfulBox(osg::Vec4(1, 1, 1, 0.8),
                                osg::Vec3(s.os.x, s.os.y, 0.0),
                                detector_.GetResolution());
      viewer_->AddChild(box);
    }
    printf("\tTook %9.3f ms to update viewer\n", t.GetMs());
  }

  for (const auto &kv : samples_per_class_) {
    printf("\t  %15s %10d samples\n", kv.first.c_str(), kv.second);
  }
}

} // namespace kitti
} // namespace app
