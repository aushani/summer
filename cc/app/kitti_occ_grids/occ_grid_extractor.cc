#include "app/kitti_occ_grids/occ_grid_extractor.h"

#include "library/timer/timer.h"

namespace app {
namespace kitti_occ_grids {

OccGridExtractor::OccGridExtractor(const std::string &save_base_fn) :
 og_builder_(150000, kResolution_, 100.0),
 camera_cal_("/home/aushani/data/kittidata/extracted/2011_09_26/"),
 rand_engine(std::chrono::system_clock::now().time_since_epoch().count()),
 save_base_path_(save_base_fn) {
  // Set size
  og_builder_.ConfigureSizeInPixels(kXYRangePixels_, kXYRangePixels_, kZRangePixels_);

  // Make sure save path exists
  if (!fs::exists(save_base_path_)) {
    printf("Making path: %s\n", save_base_path_.string().c_str());
    fs::create_directories(save_base_path_);
  }
}

void OccGridExtractor::Run() {
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

bool OccGridExtractor::ProcessLog(int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "%s/2011_09_26/2011_09_26_drive_%04d_sync/tracklet_labels.xml",
      kKittiBaseFilename, log_num);
  if (!FileExists(fn)) {
    // log doesn't exist
    return false;
  }

  kt::Tracklets tracklets;
  bool success = tracklets.loadFromFile(fn);

  if (!success) {
    return false;
  }

  printf("Loaded %d tracklets for log %d\n", tracklets.numberOfTracklets(), log_num);

  // Tracklets stats
  for (int i=0; i<tracklets.numberOfTracklets(); i++) {
    auto *tt = tracklets.getTracklet(i);
    printf("Have %s (size %5.3f x %5.3f x %5.3f) for %ld frames\n",
        tt->objectType.c_str(), tt->h, tt->w, tt->l, tt->poses.size());
  }

  // Go through velodyne for this log
  int frame = 0;
  while (ProcessFrame(&tracklets, log_num, frame)) {
    frame++;
  }

  return true;
}

std::string OccGridExtractor::GetClass(kt::Tracklets *tracklets, double x, double y, int frame) const {
  kt::Tracklets::tPose* pose = nullptr;

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
    Eigen::Vector3d x_w(x, y, 0);
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

void OccGridExtractor::ProcessFrameObjects(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame) {
  // Set up random center point in object
  double range = kResolution_*kXYRangePixels_/3;
  std::uniform_real_distribution<double> dx(-range, range);
  std::uniform_real_distribution<double> dy(-range, range);

  // Rotation
  std::uniform_real_distribution<double> theta_unif(-M_PI, M_PI);

  // Go through tracklets
  for (int t_id=0; t_id<tracklets->numberOfTracklets(); t_id++) {
    if (!tracklets->isActive(t_id, frame)) {
      continue;
    }

    auto *tt = tracklets->getTracklet(t_id);
    std::string classname = tt->objectType;
    if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background")) {
      continue;
    }

    kt::Tracklets::tPose* pose;
    tracklets->getPose(t_id, frame, pose);

    // Sample objects in different orientations and positions
    for (int i = 0; i < kObjectInstances_; i++) {
      // Get sample position
      double x = pose->tx;
      double y = pose->ty;

      // Add jitter (but skip first instance to make sure we get at least one
      // sample of object)
      if (i != 0) {
        x += dx(rand_engine);
        y += dy(rand_engine);
      }

      // Angle offset
      double dt = theta_unif(rand_engine);

      // Figure out class
      std::string label = GetClass(tracklets, x, y, frame);

      // Get occ grid
      og_builder_.SetPose(Eigen::Vector3d(x, y, 0), pose->rz + dt);
      rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

      // Save occ grid
      char fn[1000];
      sprintf(fn, "%s_%04d_%04d_%04d_%04d.og", label.c_str(), log_num, frame, t_id, i);
      fs::path path = save_base_path_ / fs::path(fn);

      DumpBin(og, path);
    }
  }
}

void OccGridExtractor::ProcessFrameBackground(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame) {
  double lower_bound = -100.0;
  double upper_bound = 100.0;
  std::uniform_real_distribution<double> xy_unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> theta_unif(-M_PI, M_PI);

  int bg_sample = 0;
  while (bg_sample < kObjectInstances_) {
    double x = xy_unif(rand_engine);
    double y = xy_unif(rand_engine);
    double t = theta_unif(rand_engine);

    // Make sure this is in camera view
    if (!camera_cal_.InCameraView(x, y, 0)) {
      continue;
    }

    // Make sure this is background
    std::string label = GetClass(tracklets, x, y, frame);

    if (label != "Background") {
      continue;
    }

    // Get occ grid
    og_builder_.SetPose(Eigen::Vector3d(x, y, 0), t);
    rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

    // Save OccGrid
    char fn[1000];
    sprintf(fn, "Background_%04d_%04d_%04d.og", log_num, frame, bg_sample);
    fs::path path = save_base_path_ / fs::path(fn);
    DumpBin(og, path);

    bg_sample++;
  }
}

bool OccGridExtractor::ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame) {
  char fn[1000];
  sprintf(fn, "%s/2011_09_26/2011_09_26_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kKittiBaseFilename, log_num, frame);

  if (!FileExists(fn)) {
    // no more scans
    return false;
  }

  library::timer::Timer t;
  kt::VelodyneScan scan(fn);
  //printf("Loaded scan %d in %5.3f sec, has %ld hits\n", frame, t.GetSeconds(), scan.GetHits().size());

  ProcessFrameObjects(tracklets, scan, log_num, frame);
  ProcessFrameBackground(tracklets, scan, log_num, frame);

  // Try to continue to the next frame
  return true;
}

bool OccGridExtractor::FileExists(const char* fn) const {
  struct stat buffer;
  return stat(fn, &buffer) == 0;
}

void OccGridExtractor::DumpBin(const rt::OccGrid &og, const fs::path &path) const {
  std::ofstream file(path.string(), std::ios::out | std::ios::binary);

  for (int i=-kXYRangePixels_; i<=kXYRangePixels_; i++) {
    for (int j=-kXYRangePixels_; j<=kXYRangePixels_; j++) {
      for (int k=-kZRangePixels_; k<=kZRangePixels_; k++) {
        float p = og.GetProbability(rt::Location(i, j, k));

        // gross
        file.write((const char*)(&p), sizeof(float));
      }
    }
  }

  file.close();
}

} // namespace kitti_occ_grids
} // namespace app
