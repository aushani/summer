#pragma once

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <random>
#include <chrono>

#include <boost/archive/binary_oarchive.hpp>

#include "library/util/angle.h"
#include "library/kitti/camera_cal.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

#include "app/kitti/model_bank.h"

namespace kt = library::kitti;
namespace ut = library::util;

namespace app {
namespace kitti {

class ModelBankBuilder {
 public:
  ModelBankBuilder();

  const ModelBank& GetModelBank() const;

 private:
  const double kZOffset_ = 0.8; // ???
  const double kPosRes_ = 0.5;
  const double kAngleRes_ = ut::DegreesToRadians(30);
  const int kEntriesPerObj_ = 10;

  ModelBank model_bank_;

  kt::Tracklets *tracklets_;
  kt::CameraCal camera_cal_;

  std::default_random_engine rand_engine;
  std::uniform_real_distribution<double> jitter_pos;
  std::uniform_real_distribution<double> jitter_angle;

  bool ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame);
  bool ProcessLog(int log_num);

  bool FileExists(const char* fn);

};

} // namespace kitti
} // namespace app
