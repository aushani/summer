#include "app/kitti_occ_grids/trainer.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "library/timer/timer.h"

namespace app {
namespace kitti_occ_grids {

Trainer::Trainer(const std::string &save_base_fn) :
 save_base_path_(save_base_fn),
 detector_(kRes_, 50, 50),
 og_builder_(200000, kRes_, 100.0),
 camera_cal_("/home/aushani/data/kittidata/extracted/2011_09_26/") {

  // Configure occ grid builder size
  og_builder_.ConfigureSizeInPixels(10, 10, 10); // +- 5 meters

  models_.insert({"Car", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Cyclist", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Pedestrian", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Background", clt::JointModel(3.0, 2.0, kRes_)});

  for (const auto &kv : models_) {
    detector_.AddModel(kv.first, kv.second);
  }

  printf("Initialized all models\n");
}

void Trainer::Run() {
  library::timer::Timer t;
  for (int log_num = 1; log_num <= 93; log_num++) {
    t.Start();
    bool res = ProcessLog(log_num);

    if (!res) {
      continue;
    }

    printf("Processed %04d in %5.3f sec\n", log_num, t.GetSeconds());
  }
}

bool Trainer::ProcessLog(int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "%s/2011_09_26/2011_09_26_drive_%04d_sync/tracklet_labels.xml",
      kKittiBaseFilename, log_num);

  if (!fs::exists(fn)) {
    return false;
  }

  kt::Tracklets tracklets;
  bool success = tracklets.loadFromFile(fn);

  if (!success) {
    return false;
  }

  printf("Loaded %d tracklets for log %d\n", tracklets.numberOfTracklets(), log_num);

  // Tracklets stats
  //for (int i=0; i<tracklets.numberOfTracklets(); i++) {
  //  auto *tt = tracklets.getTracklet(i);
  //  printf("Have %s (size %5.3f x %5.3f x %5.3f) for %ld frames\n",
  //      tt->objectType.c_str(), tt->h, tt->w, tt->l, tt->poses.size());
  //}

  // Go through velodyne for this log
  int frame = 0;
  while (ProcessFrame(&tracklets, log_num, frame)) {
    frame++;
  }

  // Save models
  std::stringstream ss;
  ss << frame;
  fs::path dir = save_base_path_ / ss.str();
  fs::create_directories(dir);
  for (const auto &kv : models_) {
    fs::path fn = dir / kv.first;
    kv.second.Save(fn.string().c_str());
  }

  return true;
}

std::string Trainer::GetTrueClass(kt::Tracklets *tracklets, int frame, const dt::ObjectState &os) const {
  kt::Tracklets::tPose* pose = nullptr;

  Eigen::Vector3d x_w(os.x, os.y, 0);

  for (int t_id=0; t_id<tracklets->numberOfTracklets(); t_id++) {
    if (!tracklets->isActive(t_id, frame)) {
      continue;
    }

    tracklets->getPose(t_id, frame, pose);
    auto tt = tracklets->getTracklet(t_id);

    Eigen::Affine3d rx(Eigen::AngleAxisd(pose->rx, Eigen::Vector3d(1, 0, 0)));
    Eigen::Affine3d ry(Eigen::AngleAxisd(pose->ry, Eigen::Vector3d(0, 1, 0)));
    Eigen::Affine3d rz(Eigen::AngleAxisd(pose->rz, Eigen::Vector3d(0, 0, 1)));
    auto r = rx * ry * rz;
    Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(pose->tx, pose->ty, pose->tz)));
    Eigen::Matrix4d X_tw = (t*r).matrix();
    Eigen::Matrix4d X_wt = X_tw.inverse();
    Eigen::Vector4d x_th = (X_wt * x_w.homogeneous());
    Eigen::Vector3d x_t = x_th.hnormalized();

    // Check if we're inside this track, otherwise this is not the track we
    // are looking for...
    if (std::fabs(x_t.x())<tt->l/2 && std::fabs(x_t.y())<tt->w/2) {
        return tt->objectType;
    }
  }

  // This is background
  return "Background";
}

std::vector<Trainer::Sample> Trainer::GetTrainingSamples(kt::Tracklets *tracklets, int frame) const {
  std::map<Sample, int> samples;

  for (double x = -detector_.GetRangeX(); x < detector_.GetRangeX(); x += detector_.GetResolution()) {
    for (double y = -detector_.GetRangeY(); y < detector_.GetRangeY(); y += detector_.GetResolution()) {
      // CHeck to make sure this is within field of view of camera and is properly labeled
      if (!camera_cal_.InCameraView(x, y, 0)) {
        continue;
      }

      dt::ObjectState os(x, y, 0);
      std::string classname = Trainer::GetTrueClass(tracklets, frame, os);

      // Is this one of the classes we care about?
      // If not, ignore for now
      if (models_.count(classname) == 0) {
        continue;
      }

      double p_class = detector_.GetProb(classname, os);

      samples[Sample(p_class, os, classname, frame)] = 0;
    }
  }

  // Find top samples
  std::vector<Sample> top_samples;
  for (const auto &kv : samples) {
    top_samples.push_back(kv.first);
    if (top_samples.size() == kSamplesPerFrame_) {
      break;
    }
  }

  return top_samples;
}

bool Trainer::ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame) {
  library::timer::Timer t;

  char fn[1000];
  sprintf(fn, "%s/2011_09_26/2011_09_26_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kKittiBaseFilename, log_num, frame);

  if (!fs::exists(fn)) {
    // no more scans
    return false;
  }

  kt::VelodyneScan scan(fn);

  // Run detector
  t.Start();
  detector_.Run(scan.GetHits());
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  // Get training samples, find out where it's more wrong
  std::vector<Sample> samples = GetTrainingSamples(tracklets, frame);

  // Update joint models
  for (const Sample &s : samples) {
    // Make occ grid
    og_builder_.SetPose(Eigen::Vector3d(s.os.x, s.os.y, 0), 0); // TODO rotation
    rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

    // Find joint model
    auto it = models_.find(s.classname);
    BOOST_ASSERT(it != models_.end());

    // Mark observations
    it->second.MarkObservations(og);
  }

  // Update detector
  for (const auto &it : models_) {
    detector_.UpdateModel(it.first, it.second);
  }

  // Try to continue to the next frame
  return true;
}

} // namespace kitti
} // namespace app
