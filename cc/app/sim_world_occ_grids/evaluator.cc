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

    jms_.insert({classname, jm});
    clts_.insert({classname, ChowLuiTree(jm)});
  }
  printf("Loaded all joint models\n");

  // Init confusion matrix
  for (const auto it1 : jms_) {
    const auto &classname1 = it1.first;
    for (const auto it2 : jms_) {
      const auto &classname2 = it2.first;
      for (int eval=0; eval<3; eval++) {
        confusion_matrix_[eval][classname1][classname2] = 0;
      }
    }
  }
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
      rt::DenseOccGrid dog(og, 5.0, 5.0, 5.0, true);

      // Classify
      for (int eval = 0; eval<3; eval++) {
        std::string best_classname = "";
        bool first = true;
        double best_log_prob = 0.0;

        t.Start();

        for (const auto it : jms_) {
          const auto &classname = it.first;
          const auto &jm = it.second;

          double log_prob = 0;

          if (eval == 0) {
            const auto it_clt = clts_.find(classname);
            BOOST_ASSERT(it_clt != clts_.end());
            log_prob = it_clt->second.EvaluateMarginalLogProbability(dog);
          } else if (eval == 1) {
            const auto it_clt = clts_.find(classname);
            BOOST_ASSERT(it_clt != clts_.end());
            log_prob = it_clt->second.EvaluateApproxLogProbability(dog);
          } else if (eval == 2) {
            ChowLuiTree clt(jm, dog);
            log_prob = clt.EvaluateLogProbability(dog);
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
        confusion_matrix_[eval][w.classname][best_classname]++;

        eval_time_ms_[eval] += ms;
        eval_counts_[eval]++;

        results_mutex_.unlock();
      }
    }
  }
}

void Evaluator::PrintResults() const {
  results_mutex_.lock();

  printf("\nMarginal Only (1-gram)\n");
  printf("Took %5.3f ms / evaluation\n", eval_time_ms_[0] / eval_counts_[0]);
  PrintConfusionMatrix(confusion_matrix_[0]);

  printf("\nApproximate CLT\n");
  printf("Took %5.3f ms / evaluation\n", eval_time_ms_[1] / eval_counts_[1]);
  PrintConfusionMatrix(confusion_matrix_[1]);

  printf("\nRebuild CLT\n");
  printf("Took %5.3f ms / evaluation\n", eval_time_ms_[2] / eval_counts_[2]);
  PrintConfusionMatrix(confusion_matrix_[2]);

  results_mutex_.unlock();
}

void Evaluator::PrintConfusionMatrix(const ConfusionMatrix &cm) const {

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

std::vector<std::string> Evaluator::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it : jms_) {
    classes.push_back(it.first);
  }

  return classes;
}

} // namespace sim_world_occ_grids
} // namespace app
