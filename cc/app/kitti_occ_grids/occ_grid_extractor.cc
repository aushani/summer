#include "app/kitti_occ_grids/occ_grid_extractor.h"

#include "library/timer/timer.h"

namespace app {
namespace kitti_occ_grids {

OccGridExtractor::OccGridExtractor(const std::string &save_base_fn) :
 og_builder_(200000, kPosRes_, 100.0),
 camera_cal_("/home/aushani/data/kittidata/extracted/2011_09_26/"),
 rand_engine(std::chrono::system_clock::now().time_since_epoch().count()),
 save_base_fn_(save_base_fn) {
  og_builder_.ConfigureSizeInPixels(10, 10, 10); // +- 3 meters
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

void OccGridExtractor::ProcessFrameObjects(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame) {
  // Go through tracklets and see which ones have hits
  for (int t_id=0; t_id<tracklets->numberOfTracklets(); t_id++) {
    if (!tracklets->isActive(t_id, frame)) {
      continue;
    }

    auto *tt = tracklets->getTracklet(t_id);
    kt::Tracklets::tPose* pose;
    tracklets->getPose(t_id, frame, pose);

    // Jitter objects according to position and angular resolution
    std::uniform_real_distribution<double> xy_unif(-kPosRes_/2, kPosRes_/2);
    std::uniform_real_distribution<double> theta_unif(-kAngleRes_/2, kAngleRes_/2);

    for (int jitter=0; jitter < kJittersPerObject_; jitter++) {
      double dx = xy_unif(rand_engine);
      double dy = xy_unif(rand_engine);
      double dt = theta_unif(rand_engine);

      og_builder_.SetPose(Eigen::Vector3d(pose->tx + dx, pose->ty + dy, 0), pose->rz + dt);
      rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

      // Save OccGrid
      char fn[1000];
      sprintf(fn, "%s/%s_%04d_%04d_%04d_%04d.og", save_base_fn_.c_str(), tt->objectType.c_str(), log_num, frame, t_id, jitter);
      og.Save(fn);
    }
  }
}

void OccGridExtractor::ProcessFrameBackground(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame) {
  int background_samples = tracklets->numberOfTracklets();

  double lower_bound = -50.0;
  double upper_bound = 50.0;
  std::uniform_real_distribution<double> xy_unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> theta_unif(-M_PI, M_PI);

  int bg_sample = 0;
  while (bg_sample < background_samples) {
    double x = xy_unif(rand_engine);
    double y = xy_unif(rand_engine);
    double t = theta_unif(rand_engine);

    // Make sure this is in camera view
    if (!camera_cal_.InCameraView(x, y, 0)) {
      continue;
    }

    // Make sure this isn't too close to any object
    bool too_close = false;
    for (int t_id=0; t_id<tracklets->numberOfTracklets(); t_id++) {
      if (!tracklets->isActive(t_id, frame)) {
        continue;
      }

      kt::Tracklets::tPose* pose;
      tracklets->getPose(t_id, frame, pose);

      if ( std::abs(pose->tx - x) < kPosRes_/2 ||
           std::abs(pose->ty - y) < kPosRes_/2) {
        too_close = true;
        break;
      }
    }

    if (too_close) {
      continue;
    }

    og_builder_.SetPose(Eigen::Vector3d(x, y, 0), t);
    rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

    char fn[1000];
    sprintf(fn, "%s/Background_%04d_%04d_%04d.og", save_base_fn_.c_str(), log_num, frame, bg_sample);
    og.Save(fn);

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

} // namespace kitti_occ_grids
} // namespace app
