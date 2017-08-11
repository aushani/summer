#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>

#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

#include "app/kitti/model_bank.h"

namespace kt = library::kitti;
namespace ut = library::util;

using namespace app::kitti;

constexpr double kZOffset = 0.8; // ???

inline bool FileExists(const char* fn) {
  struct stat buffer;
  return stat(fn, &buffer) == 0;
}

bool ProcessFrame(ModelBank *mb, kt::Tracklets *tracklets, int log_num, int frame) {
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
    ObjectState os(Eigen::Vector3d(pose->tx, pose->ty, pose->tz + kZOffset), pose->rz, tt->objectType);
    mb->MarkObservations(os, obs);
  }

  return true;
}

bool ProcessLog(ModelBank *mb, int log_num) {
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
  while (ProcessFrame(mb, &tracklets, log_num, frame)) {
    frame++;
  }

  mb->PrintStats();

  return true;
}

int main() {
  printf("Bulding Model Bank from KITTI...\n");

  ModelBank mb;

  for (int log_num = 1; log_num <= 93; log_num++) {
    bool res = ProcessLog(&mb, log_num);

    if (!res) {
      continue;
    }

    char fn[1000];
    sprintf(fn, "model_bank_%02d", log_num);
    mb.SaveModelBank(fn);
  }
}
