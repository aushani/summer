#include "app/kitti_occ_grids/trainer.h"

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

bool Trainer::ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame) {
  char fn[1000];
  sprintf(fn, "%s/2011_09_26/2011_09_26_drive_%04d_sync/velodyne_points/data/%010d.bin",
      kKittiBaseFilename, log_num, frame);

  if (!fs::exists(fn)) {
    // no more scans
    return false;
  }

  library::timer::Timer t;
  kt::VelodyneScan scan(fn);
  //printf("Loaded scan %d in %5.3f sec, has %ld hits\n", frame, t.GetSeconds(), scan.GetHits().size());

  // Run detector
  detector_.Run(scan.GetHits());

  // Get training samples, find out where it's more wrong
  // TODO

  // Update joint models
  // TODO

  // Update detector

  // Try to continue to the next frame
  return true;
}

} // namespace kitti
} // namespace app
