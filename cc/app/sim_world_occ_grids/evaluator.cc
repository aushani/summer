#include "app/sim_world_occ_grids/evaluator.h"

#include "library/ray_tracing/dense_occ_grid.h"
#include "library/timer/timer.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

namespace app {
namespace sim_world_occ_grids {

Evaluator::Evaluator(const char* training_dir, const char *testing_dir) :
 training_data_path_(training_dir), testing_data_path_(testing_dir) {

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(training_data_path_); it != end_it; it++) {
    if (fs::is_regular_file(it->path())) {
      continue;
    }

    std::string classname = it->path().filename().string();

    if (! (classname == "STAR" || classname == "BOX" || classname == "BACKGROUND")) {
      continue;
    }

    printf("Found %s\n", classname.c_str());

    fs::path p_jm = it->path() / fs::path("jm.jm");
    auto jm = JointModel::Load(p_jm.string().c_str());
    printf("Joint model is %dx%dx%d at %7.3f resolution\n",
        jm.GetNXY(), jm.GetNXY(), jm.GetNZ(), jm.GetResolution());

    jms_.insert({classname, jm});
    clts_.insert({classname, ChowLuiTree(jm)});
  }
  printf("Loaded all joint models\n");
}

Evaluator::~Evaluator() {
  Finish();
}

void Evaluator::QueueClass(const std::string &classname) {
  work_queue_mutex_.lock();

  fs::path p_class = testing_data_path_ / fs::path(classname);

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p_class); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension() != ".og") {
      continue;
    }

    work_queue_.push_back(Work(it->path(), classname));
  }

  work_queue_mutex_.unlock();
}

void Evaluator::QueueEvalType(const std::string &type_str) {
  if (type_str == "DENSE") {
    QueueEvalType(ChowLuiTree::DENSE);
  } else if (type_str == "APPROX_MARGINAL") {
    QueueEvalType(ChowLuiTree::APPROX_MARGINAL);
  } else if (type_str == "APPROX_CONDITIONAL") {
    QueueEvalType(ChowLuiTree::APPROX_CONDITIONAL);
  } else if (type_str == "APPROX_GREEDY") {
    QueueEvalType(ChowLuiTree::APPROX_GREEDY);
  } else if (type_str == "MARGINAL") {
    QueueEvalType(ChowLuiTree::MARGINAL);
  } else {
    BOOST_ASSERT(false);
  }
}

void Evaluator::QueueEvalType(const ChowLuiTree::EvalType &type) {
  work_queue_mutex_.lock();
  results_mutex_.lock();

  // Init results matrix
  results_[type] = Results();

  results_mutex_.unlock();
  work_queue_mutex_.unlock();
}

bool Evaluator::HaveWork() const {
  bool res = false;

  work_queue_mutex_.lock();
  res = work_queue_.size();
  work_queue_mutex_.unlock();

  return res;
}

void Evaluator::Start() {
  printf("Have %ld og's to evaluate\n", work_queue_.size());

  // Shuffle
  work_queue_mutex_.lock();
  std::random_shuffle(work_queue_.begin(), work_queue_.end());
  work_queue_mutex_.unlock();

  for (int i=0; i<kNumThreads_; i++) {
    threads_.push_back(std::thread(&Evaluator::WorkerThread, this));
  }
}

void Evaluator::Finish() {
  for (auto &t : threads_) {
    t.join();
  }
}

void Evaluator::WorkerThread() {
  library::timer::Timer t;
  bool done = false;

  while (!done) {
    work_queue_mutex_.lock();
    if (work_queue_.empty()) {
      done = true;
      work_queue_mutex_.unlock();
    } else {
      // Get work
      Work w = work_queue_.front();
      work_queue_.pop_front();
      work_queue_mutex_.unlock();

      //Process Work
      auto og = rt::OccGrid::Load(w.path.string().c_str());
      rt::DenseOccGrid dog(og, 5.0, 5.0, 1.0, true); // clamp to binary

      // Classify
      for (auto &it_results : results_) {
        const auto &type = it_results.first;
        auto &results = it_results.second;

        std::string best_classname = "";
        bool first = true;
        double best_log_prob = 0.0;

        t.Start();

        for (const auto &it : jms_) {
          const auto &classname = it.first;
          const auto &jm = it.second;

          double log_prob = 0;

          if (type == ChowLuiTree::DENSE) {
            // Have to rebuild CLT
            ChowLuiTree clt(jm, dog);
            log_prob = clt.EvaluateLogProbability(dog, type);
          } else {
            const auto &it_clt = clts_.find(classname);
            BOOST_ASSERT(it_clt != clts_.end());

            log_prob = it_clt->second.EvaluateLogProbability(dog, type);
          }

          if (first || log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_classname = classname;
          }

          first = false;
        }
        double ms = t.GetMs();

        // Add to results
        results_mutex_.lock();

        results.CountTime(ms);
        results.MarkResult(w.classname, best_classname);

        results_mutex_.unlock();
      }
    }
  }
}

void Evaluator::PrintResults() const {
  results_mutex_.lock();

  for (const auto &it_results : results_) {
    printf("\n%s\n", GetEvalTypeString(it_results.first).c_str());
    it_results.second.Print();
  }

  results_mutex_.unlock();
}

std::vector<std::string> Evaluator::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it : jms_) {
    classes.push_back(it.first);
  }

  return classes;
}

std::string Evaluator::GetEvalTypeString(const ChowLuiTree::EvalType &type) const {
  if (type == ChowLuiTree::DENSE) {
    return "DENSE";
  }

  if (type == ChowLuiTree::APPROX_MARGINAL) {
    return "APPROX_MARGINAL";
  }

  if (type == ChowLuiTree::APPROX_CONDITIONAL) {
    return "APPROX_CONDITIONAL";
  }

  if (type == ChowLuiTree::APPROX_GREEDY) {
    return "APPROX_GREEDY";
  }

  if (type == ChowLuiTree::MARGINAL) {
    return "MARGINAL";
  }

  BOOST_ASSERT(false);
  return "invalid";
}

} // namespace sim_world_occ_grids
} // namespace app
