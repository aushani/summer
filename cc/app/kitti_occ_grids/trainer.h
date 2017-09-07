#pragma once

#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>

#include "library/kitti/camera_cal.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/detector/detector.h"

namespace fs = boost::filesystem;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace dt = library::detector;
namespace clt = library::chow_liu_tree;

namespace app {
namespace kitti_occ_grids {

class Trainer {
 public:
  Trainer(const std::string &save_base_fn);

  void Run();

 private:
  const char* kKittiBaseFilename = "/home/aushani/data/kittidata/extracted/";
  static constexpr double kRes_ = 0.50;                    // 50 cm
  static constexpr int kAngleBins_ = 8;                       // 8 angle bins -> 45 degrees
  static constexpr double kAngleRes_ = 2 * M_PI/kAngleBins_;  // 45 degrees
  static constexpr int kJittersPerObject_ = 10;

  fs::path save_base_path_;

  dt::Detector detector_;
  rt::OccGridBuilder og_builder_;
  kt::CameraCal camera_cal_;

  std::map<std::string, clt::JointModel> models_;

  bool ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame);
  bool ProcessLog(int log_num);

  bool FileExists(const char* fn) const;
};

} // namespace kitti_occ_grids
} // namespace app
