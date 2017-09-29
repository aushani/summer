#include "app/kitti_occ_grids/trainer.h"

#include <algorithm>

#include <boost/format.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "library/osg_nodes/colorful_box.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/osg_nodes/object_labels.h"
#include "library/timer/timer.h"
#include "library/util/angle.h"

#include "app/kitti_occ_grids/map_node.h"

namespace osgn = library::osg_nodes;
namespace ut = library::util;

namespace app {
namespace kitti_occ_grids {

Trainer::Trainer(const std::string &save_base_fn) :
 save_base_path_(save_base_fn),
 detector_(dt::Dim(0, 75, -50, 50, kRes_)),
 og_builder_(150000, kRes_, 75.0),
 model_bank_(3.0, 2.0, kRes_) {

  // Configure occ grid builder size
  og_builder_.ConfigureSize(3.0, 3.0, 2.0);

  model_bank_.AddClass("Car", 8);
  model_bank_.AddClass("Cyclist", 8);
  model_bank_.AddClass("Pedestrian", 1);
  model_bank_.AddClass("Background", 1);

  detector_.SetModelBank(model_bank_);
  printf("Initialized model bank\n");

  const int width = detector_.GetDim().n_x;
  const int height = detector_.GetDim().n_y;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      dt::ObjectState os = detector_.GetDim().GetState(i, j);
      states_.emplace_back(os);
    }
  }

  printf("Have %ld states\n", states_.size());
}

void Trainer::LoadFrom(const std::string &load_fn) {
  model_bank_ = ft::ModelBank::Load(load_fn.c_str());
  detector_.UpdateModelBank(model_bank_);
}

void Trainer::SetViewer(const std::shared_ptr<vw::Viewer> &viewer) {
  viewer_ = viewer;
}

void Trainer::Run(int first_epoch, int first_frame) {
  int epoch = first_epoch;
  int starting_frame = first_frame;

  while (true) {
    library::timer::Timer t;
    for (int frame = starting_frame; frame < kNumFrames; frame++) {
      t.Start();
      ProcessFrame(frame);
      printf("  Processed frame %04d in %9.3f sec\n", frame, t.GetSeconds());

      // Save?
      if (frame % 100 == 0) {
        t.Start();

        fs::path path = save_base_path_ / (boost::format("%|04|_%|06|.mb") % epoch % frame).str();
        fs::create_directories(save_base_path_);

        model_bank_.Save(path.string().c_str());
        printf("Took %9.3f ms to save model bank\n", t.GetMs());
      }
    }

    epoch++;
    starting_frame = 0;
  }
}

void Trainer::RunBackground(int first_epoch, int first_frame_num) {
  run_thread_ = std::thread(&Trainer::Run, this, first_epoch, first_frame_num);
}

std::string Trainer::GetTrueClass(const kt::KittiChallengeData &kcd, const dt::ObjectState &os, double *theta) const {
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

    // TODO inside box or just within kRes_ of center????

    double dl = std::fabs(x_object.x()); // Yes, z
    double dw = std::fabs(x_object.z());
    if (dl<kRes_/2 && dw<kRes_/2) {
      if (theta) {
        *theta = label.rotation_y;
      }
      return kt::ObjectLabel::GetString(label.type);
    }

    // Add a resolution's worth of buffer
    double dl_buffer = std::fabs(x_object.x()) - kRes_; // Yes, z
    double dw_buffer = std::fabs(x_object.z()) - kRes_;
    if (dl_buffer<length/2 && dw_buffer<width/2) {
      return "Closeish"; // XXX
    }

  }

  // This is background
  return "Background";
}

void Trainer::GetTrainingSamplesWorker(const kt::KittiChallengeData &kcd, size_t idx0, size_t idx1,
    std::map<std::string, std::vector<Sample> > *samples, std::map<std::string, double> *total_weight, std::mutex *mutex) const {
  std::map<std::string, std::vector<Sample> > my_samples;
  std::map<std::string, double > my_total_weight;

  for (size_t idx = idx0; idx < idx1; idx++) {
    auto os = states_[idx];

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

    double theta = 0;
    std::string classname = GetTrueClass(kcd, os, &theta);

    // Is this one of the classes we care about?
    // If not, ignore for now
    const auto &classes = model_bank_.GetClasses();
    if (std::find(classes.begin(), classes.end(), classname) == classes.end()) {
      continue;
    }

    double p_class = detector_.GetProb(classname, os);

    Sample s(p_class, os, classname, theta);
    my_samples[classname].push_back(s);
    my_total_weight[classname] += s.p_wrong;
  }

  mutex->lock();
  //printf("Thread--- \n");
  for (const auto &kv : my_samples) {
    //printf("\tHave %ld samples for %s\n", kv.second.size(), kv.first.c_str());
    for (const auto &s : kv.second) {
      (*samples)[kv.first].push_back(s);
    }
    (*total_weight)[kv.first] += my_total_weight[kv.first];
  }
  mutex->unlock();
}

std::vector<Trainer::Sample> Trainer::GetTrainingSamples(const kt::KittiChallengeData &kcd) const {
  std::map<std::string, std::vector<Sample> > samples;
  std::map<std::string, double > total_weight;

  std::mutex mutex;

  std::vector<std::thread> threads;
  int n_threads = 6;
  for (int i=0; i<n_threads; i++) {
    size_t idx0 = i * states_.size() / n_threads;
    size_t idx1 = (i+1) * states_.size() / n_threads;
    threads.emplace_back(&Trainer::GetTrainingSamplesWorker, this, kcd, idx0, idx1, &samples, &total_weight, &mutex);
  }

  for (auto &t : threads) {
    t.join();
  }

  std::vector<Sample> chosen_samples;

  for (auto &kv : samples) {
    std::vector<Sample> &class_samples = kv.second;
    if (class_samples.size() < kSamplesPerFrame_) {
      for (const auto &s : class_samples) {
      chosen_samples.push_back(s);
      }
    } else {
      std::random_shuffle(class_samples.begin(), class_samples.end());

      double weight_rollover = total_weight[kv.first] / kSamplesPerFrame_;
      double weight_at = 0.0;
      for (const auto &s : class_samples) {
        weight_at += s.p_wrong;
        if (weight_at > weight_rollover) {
          weight_at -= weight_rollover;
          chosen_samples.push_back(s);
        }
      }
    }
  }


  return chosen_samples;
}

void Trainer::UpdateViewer(Trainer *trainer, const kt::KittiChallengeData &kcd, const std::vector<Sample> &samples) {
  if (trainer->viewer_) {
    library::timer::Timer t;
    osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(kcd.GetScan());
    osg::ref_ptr<osgn::ObjectLabels> labels = new osgn::ObjectLabels(kcd.GetLabels(), kcd.GetTcv());
    osg::ref_ptr<MapNode> map_node = new MapNode(trainer->detector_, kcd);

    trainer->viewer_->RemoveAllChildren();
    trainer->viewer_->AddChild(pc);
    trainer->viewer_->AddChild(labels);
    trainer->viewer_->AddChild(map_node);

    // Add samples
    for (const Sample &s : samples) {
      osg::ref_ptr<osgn::ColorfulBox> box
        = new osgn::ColorfulBox(osg::Vec4(1, 1, 1, 0.8),
                                osg::Vec3(s.os.x, s.os.y, 0.0),
                                trainer->detector_.GetDim().res);
      trainer->viewer_->AddChild(box);
    }
    printf("\tTook %9.3f ms to update viewer\n", t.GetMs());
  }
}

void Trainer::Train(Trainer *trainer, const kt::KittiChallengeData &kcd, const std::vector<Sample> &samples) {
  library::timer::Timer t;

  // Update models in detector
  for (const Sample &s : samples) {
    int angle_bins = trainer->model_bank_.GetNumAngleBins(s.classname);
    double angle_res = 2*M_PI / angle_bins;

    for (int angle_bin = 0; angle_bin < angle_bins; angle_bin++) {
      // Make occ grid
      trainer->og_builder_.SetPose(Eigen::Vector3d(s.os.x, s.os.y, 0), s.theta - angle_bin * angle_res); // TODO jitter?
      auto fog = trainer->og_builder_.GenerateFeatureOccGrid(kcd.GetScan().GetHits(), kcd.GetScan().GetIntensities());

      auto &model = trainer->model_bank_.GetFeatureModel(s.classname, angle_bin);

      model.MarkObservations(fog);
      trainer->samples_per_class_[s.classname]++;
    }
  }
  printf("\tTook %9.3f ms to update models\n", t.GetMs());
}

void Trainer::ProcessFrame(int frame) {
  library::timer::Timer t;

  // Get data
  kt::KittiChallengeData kcd = kt::KittiChallengeData::LoadFrame(kKittiBaseFilename, frame);

  // Run detector
  t.Start();
  detector_.Run(kcd.GetScan().GetHits(), kcd.GetScan().GetIntensities());
  printf("\tTook %9.3f ms to run detector\n", t.GetMs());

  // Get training samples, find out where it's more wrong
  t.Start();
  std::vector<Sample> samples = GetTrainingSamples(kcd);
  printf("\tTook %9.3f ms to get %ld training samples\n", t.GetMs(), samples.size());

  // Start threads
  std::thread train_thread(&Trainer::Train, this, kcd, samples);
  std::thread viewer_thread(&Trainer::UpdateViewer, this, kcd, samples);

  viewer_thread.join();
  train_thread.join();

  for (const auto &kv : samples_per_class_) {
    printf("\t  %15s %10d total samples so far\n", kv.first.c_str(), kv.second);
  }
}

} // namespace kitti
} // namespace app
