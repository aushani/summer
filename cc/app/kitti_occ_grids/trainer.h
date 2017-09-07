#pragma once

#include <iostream>
#include <fstream>
#include <set>

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
  struct Sample {
    double p_correct = 0;
    dt::ObjectState os;
    std::string classname;
    int frame_num = 0;

    Sample(double pc, const dt::ObjectState s, const std::string &cn, int f) :
     p_correct(pc), os(s), classname(cn), frame_num(f) {
      my_count_ = count_++;
    };

    bool operator<(const Sample &s) const {
      if (p_correct != s.p_correct) {
        return p_correct < s.p_correct;
      }

      return count_ < s.count_;
    }

   private:
    static int count_;
    int my_count_ = 0;
  };

  const char* kKittiBaseFilename = "/home/aushani/data/kittidata/extracted/";
  static constexpr double kRes_ = 0.50;                    // 50 cm
  static constexpr int kSamplesPerClassPerFrame_ = 25;

  fs::path save_base_path_;

  dt::Detector detector_;
  rt::OccGridBuilder og_builder_;
  kt::CameraCal camera_cal_;

  std::map<std::string, clt::JointModel> models_;

  std::vector<dt::ObjectState> states_;

  bool ProcessFrame(kt::Tracklets *tracklets, int log_num, int frame);
  bool ProcessLog(int log_num);

  bool FileExists(const char* fn) const;

  std::string GetTrueClass(kt::Tracklets *tracklets, int frame, const dt::ObjectState &os) const;
  std::multiset<Sample> GetTrainingSamples(kt::Tracklets *tracklets, int frame) const;
};

} // namespace kitti_occ_grids
} // namespace app