#pragma once

#include <thread>
#include <mutex>
#include <queue>

#include <boost/filesystem.hpp>

#include "app/sim_world_occ_grids/chow_lui_tree.h"
#include "app/sim_world_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

namespace app {
namespace sim_world_occ_grids {

class Evaluator {
 public:
  Evaluator(const char* training_dir, const char* testing_dir);
  ~Evaluator();

  void QueueClass(const std::string &classname);

  void Start();
  void Finish();

  bool HaveWork() const;
  void PrintResults() const;

  std::vector<std::string> GetClasses() const;

 private:
  static constexpr int kNumThreads_ = 48;
  static constexpr int kEvals_ = 4;

  fs::path training_data_path_;
  fs::path testing_data_path_;

  std::map<std::string, ChowLuiTree> clts_;
  std::map<std::string, JointModel> jms_;

  typedef std::map<std::string, std::map<std::string, int> > ConfusionMatrix;
  ConfusionMatrix confusion_matrix_[kEvals_];

  mutable std::mutex results_mutex_;

  struct Work {
    fs::path path;
    std::string classname;

    Work(const fs::path &p, const std::string &cn) : path(p), classname(cn) { }
  };

  std::deque<Work> work_queue_;
  mutable std::mutex work_queue_mutex_;

  std::vector<std::thread> threads_;

  double eval_time_ms_[kEvals_] = {0.0, 0.0, 0.0, 0.0};
  int eval_counts_[kEvals_] = {0, 0, 0, 0};

  void WorkerThread();
  void PrintConfusionMatrix(const ConfusionMatrix &cm) const;
};

} // namespace sim_world_occ_grids
} // namespace app

