#include "app/kitti/model_bank_builder.h"

namespace app {
namespace kitti {

ModelBankBuilder::ModelBankBuilder() :
 camera_cal_("/home/aushani/data/kittidata/extracted/2011_09_26/"),
 rand_engine(std::chrono::system_clock::now().time_since_epoch().count()),
 jitter_pos(-kPosRes_/2, kPosRes_/2),
 jitter_angle(-kAngleRes_/2, kAngleRes_/2) {
  for (int log_num = 1; log_num <= 93; log_num++) {
    bool res = ProcessLog(log_num);

    if (!res) {
      continue;
    }

    char fn[1000];
    sprintf(fn, "model_bank_%02d", log_num);
    model_bank_.SaveModelBank(fn);
  }
}

bool ModelBankBuilder::ProcessLog(int log_num) {
  // Load Tracklets
  char fn[1000];
  sprintf(fn, "/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_%04d_sync/tracklet_labels.xml", log_num);
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

  model_bank_.PrintStats();

  return true;
}

bool ModelBankBuilder::ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame) {
  char fn[1000];
  sprintf(fn, "/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_%04d_sync/velodyne_points/data/%010d.bin", log_num, frame);

  if (!FileExists(fn)) {
    // no more scans
    return false;
  }

  kt::VelodyneScan scan(fn);
  printf("Loaded scan %d, has %ld hits\n", frame, scan.GetHits().size());

  // Convert Eigen::Vector3d to Observations
  std::vector<Observation> obs;
  for (const auto &hit: scan.GetHits()) {
    obs.emplace_back(hit);
  }

  // Go through tracklets and see which ones have hits
  for (int t_id=0; t_id<tracklets->numberOfTracklets(); t_id++) {
    if (!tracklets->isActive(t_id, frame)) {
      continue;
    }

    auto *tt = tracklets->getTracklet(t_id);
    kt::Tracklets::tPose* pose;
    tracklets->getPose(t_id, frame, pose);

    // Mark observations for state
    for (int i=0; i<kEntriesPerObj_; i++) {
      double dx = jitter_pos(rand_engine);
      double dy = jitter_pos(rand_engine);
      double dz = jitter_pos(rand_engine);

      double dt = jitter_angle(rand_engine);

      ObjectState os(Eigen::Vector3d(pose->tx + dx, pose->ty + dy, pose->tz + kZOffset_ + dz), pose->rz + dt, tt->objectType);
      model_bank_.MarkObservations(os, obs);
    }
  }

  // Sample background
  int background_samples = tracklets->numberOfTracklets() * kEntriesPerObj_;
  double lower_bound = -20.0;
  double upper_bound = 20.0;
  std::uniform_real_distribution<double> xy_unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> z_unif(0, 2);

  for (int i=0; i<background_samples; i++) {
    double x = xy_unif(rand_engine);
    double y = xy_unif(rand_engine);
    double z = z_unif(rand_engine);

    // Make sure this is in camera view
    if (!camera_cal_.InCameraView(x, y, z)) {
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
           std::abs(pose->ty - y) < kPosRes_/2 ||
           std::abs(pose->tz - z) < kPosRes_/2) {
        too_close = true;
        break;
      }
    }

    if (too_close) {
      continue;
    }

    // Mark observations for no obj.
    ObjectState os(Eigen::Vector3d(x, y, z + kZOffset_), 0, "NOOBJ");
    model_bank_.MarkObservations(os, obs);
  }

  return true;
}

const ModelBank& ModelBankBuilder::GetModelBank() const {
  return model_bank_;
}

bool ModelBankBuilder::FileExists(const char* fn) {
  struct stat buffer;
  return stat(fn, &buffer) == 0;
}

} // namespace kitti
} // namespace app
