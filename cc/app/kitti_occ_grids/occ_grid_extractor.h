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
  static constexpr double kResolution_ = 0.30;                // 30 cm
  static constexpr int kXYRangePixels_ = 16*3;                // +- 4.8 * 3 m
  static constexpr int kZRangePixels_ = 16;                   // +- 4.8 m
  static constexpr int kObjectInstances_ = 10;

  rt::OccGridBuilder og_builder_;
  kt::CameraCal camera_cal_;
  std::default_random_engine rand_engine;

  fs::path save_base_path_;

  void ProcessFrameObjects(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame);
  void ProcessFrameBackground(kt::Tracklets *tracklets, const kt::VelodyneScan &scan, int log_num, int frame);

  bool ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame);
  bool ProcessLog(int log_num);

  bool FileExists(const char* fn) const;

  void DumpBin(const rt::OccGrid &og, const fs::path &path) const;

  std::string GetClass(kt::Tracklets *tracklets, double x, double y, int frame) const;
};

} // namespace kitti
} // namespace app
