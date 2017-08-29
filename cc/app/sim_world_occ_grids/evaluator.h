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
  void QueueEvalType(const ChowLuiTree::EvalType &type);
  void QueueEvalType(const std::string &string);

  void Start();
  void Finish();

  bool HaveWork() const;
  void PrintResults() const;

  std::vector<std::string> GetClasses() const;

 private:
  static constexpr int kNumThreads_ = 48;

  fs::path training_data_path_;
  fs::path testing_data_path_;

  std::map<std::string, ChowLuiTree> clts_;
  std::map<std::string, JointModel> jms_;

  struct Work {
    fs::path path;
    std::string classname;

    Work(const fs::path &p, const std::string &cn) : path(p), classname(cn) { }
  };

  std::deque<Work> work_queue_;
  mutable std::mutex work_queue_mutex_;

  std::vector<std::thread> threads_;

  struct Results {
    std::map<std::string, std::map<std::string, int> > cm;

    double time_ms;
    int count;

    double AverageTime() const {
      return time_ms / count;
    }

    void CountTime(double t) {
      time_ms += t;
      count++;
    }

    void MarkResult(const std::string &true_class, const std::string &labeled_class) {
      cm[true_class][labeled_class]++;
    }

    void Print() const {
      printf("Took %5.3f ms / evaluation\n", AverageTime());

      printf("%15s  ", "");
      for (const auto it1 : cm) {
        const auto &classname1 = it1.first;
        printf("%15s  ", classname1.c_str());
      }
      printf("\n");

      for (const auto it1 : cm) {
        const auto &classname1 = it1.first;
        printf("%15s  ", classname1.c_str());

        int sum = 0;
        for (const auto it2 : it1.second) {
          sum += it2.second;
        }

        for (const auto it2 : it1.second) {
          printf("%15.1f %%", sum == 0 ? 0:(100.0 * it2.second/sum));
        }
        printf("\t(%6d Evaluations)\n", sum);
      }
    }
  };

  std::map<ChowLuiTree::EvalType, Results> results_;
  mutable std::mutex results_mutex_;

  void WorkerThread();
  std::string GetEvalTypeString(const ChowLuiTree::EvalType &type) const;
};

} // namespace sim_world_occ_grids
} // namespace app

