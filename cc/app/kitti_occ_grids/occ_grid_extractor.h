#pragma once

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <random>
#include <chrono>

#include <boost/filesystem.hpp>

#include "library/kitti/camera_cal.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/ray_tracing/occ_grid_builder.h"

namespace fs = boost::filesystem;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class OccGridExtractor {
 public:
  OccGridExtractor(const std::string &save_base_fn);

  void Run();

 private:
  const char* kKittiBaseFilename = "/home/aushani/data/kittidata/extracted/";
  static constexpr double kPosRes_ = 0.50;                    // 50 cm
  static constexpr int kAngleBins_ = 1;                       // 8 angle bins -> 45 degrees
  static constexpr double kAngleRes_ = 2 * M_PI/kAngleBins_;  // 45 degrees
  static constexpr int kJittersPerObject_ = 10;

  rt::OccGridBuilder og_builder_;
  kt::CameraCal camera_cal_;
  std::default_random_engine rand_engine;

  fs::path save_base_path_;

  void ProcessFrameObjects(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame);
  void ProcessFrameBackground(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame);

  bool ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame);
  bool ProcessLog(int log_num);

  bool FileExists(const char* fn) const;
};

} // namespace kitti
} // namespace app
