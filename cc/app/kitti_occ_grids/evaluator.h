#pragma once

#include <thread>
#include <mutex>
#include <queue>

#include <boost/filesystem.hpp>

#include "app/kitti_occ_grids/dynamic_clt.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

namespace app {
namespace kitti_occ_grids {

class Evaluator {
 public:
  Evaluator(const char* training_dir, const char* testing_dir);
  ~Evaluator();

  void QueueClass(const std::string &classname);

  void Start();
  void Finish();

  bool HaveWork() const;
  size_t WorkLeft() const;
  void PrintConfusionMatrix() const;

  const std::vector<std::string>& GetClasses() const;

 private:
  static constexpr int kNumThreads_ = 1;

  fs::path training_data_path_;
  fs::path testing_data_path_;

  //std::map<std::string, ChowLuiTree> clts_;
  //std::map<std::string, JointModel> jms_;
  std::vector<DynamicCLT*> dclts_;
  std::vector<std::string> classes_;

  std::map<std::string, std::map<std::string, int> > confusion_matrix_;
  mutable std::mutex results_mutex_;

  struct Work {
    fs::path path;
    std::string classname;

    Work(const fs::path &p, const std::string &cn) : path(p), classname(cn) { }
  };

  std::map<std::string, std::deque<Work> > work_queues_;
  mutable std::mutex work_queue_mutex_;

  std::vector<std::thread> threads_;

  double load_time_ms_ = 0;
  double dog_time_ms_ = 0;
  double eval_time_ms_ = 0;
  double total_time_ms_ = 0;
  int timing_counts_ = 0;

  void WorkerThread();
  void ProcessWork(const Work &w);
};

} // namespace kitti_occ_grids
} // namespace app

