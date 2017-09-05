#include "app/sim_world_occ_grids/evaluator.h"

#include "library/ray_tracing/dense_occ_grid.h"
#include "library/timer/timer.h"

namespace clt = library::chow_liu_tree;
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
    const clt::JointModel jm = clt::JointModel::Load(p_jm.string().c_str());
    printf("Joint model is %ldx%ldx%ld at %7.3f resolution\n",
        jm.GetNXY(), jm.GetNXY(), jm.GetNZ(), jm.GetResolution());

    clts_.insert({classname, clt::DynamicCLT(jm)});
    jms_.insert({classname, jm});
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
      rt::DenseOccGrid dog(og, 5.0, 5.0, 0.0, true); // clamp to binary

      // Classify
      std::string best_classname = "";
      bool first = true;
      double best_log_prob = 0.0;

      t.Start();
      //printf("%s\n", w.classname.c_str());

      for (const auto &it : clts_) {
        const auto &classname = it.first;
        const auto &clt = it.second;

        //double log_prob = clt.BuildAndEvaluate(dog);
        //double log_prob = clt.EvaluateMarginal(dog);
        double log_prob = clt.BuildAndEvaluateGreedy(og);

        if (first || log_prob > best_log_prob) {
          best_log_prob = log_prob;
          best_classname = classname;
        }

        first = false;
      }
      double ms = t.GetMs();
      //printf("\tbest score for %s is %s (%f)\n", w.classname.c_str(), best_classname.c_str(), best_log_prob);

      // Add to results
      results_.CountTime(ms);
      results_.MarkResult(w.classname, best_classname);
    }
  }
}

void Evaluator::PrintResults() const {
  printf("Result:\n");
  results_.Print();
}

std::vector<std::string> Evaluator::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it : clts_) {
    classes.push_back(it.first);
  }

  return classes;
}

} // namespace sim_world_occ_grids
} // namespace app
