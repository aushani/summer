#pragma once

#include <iostream>
#include <fstream>
#include <set>
#include <thread>

#include <boost/filesystem.hpp>

#include "library/kitti/kitti_challenge_data.h"
#include "library/kitti/velodyne_scan.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/detector/detector.h"
#include "library/viewer/viewer.h"

namespace fs = boost::filesystem;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace dt = library::detector;
namespace vw = library::viewer;
namespace clt = library::chow_liu_tree;

namespace app {
namespace kitti_occ_grids {

class Trainer {
 public:
  Trainer(const std::string &save_base_fn);

  void LoadFrom(const std::string &load_base_dir);

  void SetViewer(const std::shared_ptr<vw::Viewer> &viewer);
  void Run(int first_epoch = 0, int first_frame_num=0);
  void RunBackground(int first_epoch = 0, int first_frame_num=0);

 private:
  struct Sample {
    double p_correct = 0;
    double p_wrong = 0;
    dt::ObjectState os;
    std::string classname;

    Sample(double pc, const dt::ObjectState s, const std::string &cn) :
     p_correct(pc), p_wrong(1-pc), os(s), classname(cn) {
    };

    bool operator<(const Sample &s) const {
      if (p_correct != s.p_correct) {
        return p_correct < s.p_correct;
      }
    }
  };

  const char* kKittiBaseFilename = "/home/aushani/data/kitti_challenge/";
  const int kNumFrames = 7481;
  static constexpr double kRes_ = 0.50;                    // 50 cm
  static constexpr int kSamplesPerFrame_ = 10;

  fs::path save_base_path_;

  std::shared_ptr<vw::Viewer> viewer_;

  dt::Detector detector_;
  rt::OccGridBuilder og_builder_;

  std::map<std::string, clt::JointModel> models_;
  std::map<std::string, int> samples_per_class_;

  std::vector<dt::ObjectState> states_;

  std::thread run_thread_;

  void ProcessFrame(int frame);

  std::string GetTrueClass(const kt::KittiChallengeData &kcd, const dt::ObjectState &os) const;
  std::vector<Sample> GetTrainingSamples(const kt::KittiChallengeData &kcd) const;

  static void Train(Trainer *trainer, const kt::KittiChallengeData &kcd, const std::vector<Sample> &samples);
  static void UpdateViewer(Trainer *trainer, const kt::KittiChallengeData &kcd, const std::vector<Sample> &samples);
};

} // namespace kitti_occ_grids
} // namespace app
