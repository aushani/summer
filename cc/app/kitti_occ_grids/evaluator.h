#pragma once

#include <thread>
#include <mutex>
#include <queue>

#include <boost/filesystem.hpp>

#include "app/kitti_occ_grids/chow_lui_tree.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

namespace app {
namespace kitti_occ_grids {

class Evaluator {
 public:
  Evaluator(const char* training_dir, const char* testing_dir, const ChowLuiTree::EvalType &type);
  ~Evaluator();

  void QueueClass(const std::string &classname);

  void Start();
  void Finish();

  bool HaveWork() const;
  void PrintConfusionMatrix() const;

  std::vector<std::string> GetClasses() const;

 private:
  static constexpr int kNumThreads_ = 10;

  fs::path training_data_path_;
  fs::path testing_data_path_;

  ChowLuiTree::EvalType eval_type_;

  std::map<std::string, ChowLuiTree> clts_;
  std::map<std::string, std::map<std::string, int> > confusion_matrix_;
  mutable std::mutex results_mutex_;

  struct Work {
    fs::path path;
    std::string classname;

    Work(const fs::path &p, const std::string &cn) : path(p), classname(cn) { }
  };

  std::deque<Work> work_queue_;
  mutable std::mutex work_queue_mutex_;

  std::vector<std::thread> threads_;

  double eval_time_ms_ = 0;
  int eval_counts_ = 0;

  void WorkerThread();
};

} // namespace kitti_occ_grids
} // namespace app

