#include "app/kitti_occ_grids/evaluator.h"

#include "library/timer/timer.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

namespace app {
namespace kitti_occ_grids {

Evaluator::Evaluator(const char* training_dir, const char *testing_dir) :
 training_data_path_(training_dir), testing_data_path_(testing_dir) {

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(training_data_path_); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension() != ".mm") {
      continue;
    }

    std::string classname = it->path().stem().string();

    if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background") ) {
    //if (! (classname == "Car" || classname == "Background") ) {
      continue;
    }

    printf("Found %s\n", classname.c_str());

    //fs::path p_clf = it->path() / fs::path("clt.clt");
    //auto clt = ChowLuiTree::Load(p_clf.string().c_str());
    //printf("CLT size: %ld\n", clt.Size());
    //clts_.insert({classname, clt});

    library::timer::Timer t;
    auto mm = MarginalModel::Load(it->path().string().c_str());
    printf("\tTook %5.3f sec to load %s\n", t.GetSeconds(), it->path().string().c_str());
    mms_.insert({classname, mm});
  }
  printf("Loaded %ld models\n", mms_.size());

  // Init confusion matrix
  for (const auto it1 : mms_) {
    const auto &classname1 = it1.first;
    for (const auto it2 : mms_) {
      const auto &classname2 = it2.first;
      confusion_matrix_[classname1][classname2] = 0;
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

    work_queues_[classname].push_back(Work(it->path(), classname));
  }

  work_queue_mutex_.unlock();
}

bool Evaluator::HaveWork() const {
  return WorkLeft() > 0;
}

size_t Evaluator::WorkLeft() const {
  size_t count = 0;

  work_queue_mutex_.lock();
  for (const auto &it : work_queues_) {
    count += it.second.size();
  }
  work_queue_mutex_.unlock();

  return count;
}

void Evaluator::Start() {
  printf("Have %ld og's to evaluate\n", WorkLeft());

  // Shuffle
  work_queue_mutex_.lock();
  for (auto &it : work_queues_) {
    std::random_shuffle(it.second.begin(), it.second.end());
  }
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
  size_t count = 0;

  while (HaveWork()) {
    // Get work

    size_t num_queues = work_queues_.size();
    size_t queue_num = count % num_queues;

    auto it = work_queues_.begin();
    std::advance(it, queue_num);
    auto &work_queue = it->second;

    work_queue_mutex_.lock();
    if (work_queue.size() > 0) {
      Work w = work_queue.front();
      work_queue.pop_front();

      work_queue_mutex_.unlock();

      //Process Work
      ProcessWork(w);
    } else {
      work_queue_mutex_.unlock();
    }

    count++;
  }
}

void Evaluator::ProcessWork(const Work &w) {
  library::timer::Timer t_total;

  //printf("loading %s\n", w.path.string().c_str());
  auto og = rt::OccGrid::Load(w.path.string().c_str());

  // Classify
  std::string best_classname = "";
  bool first = true;
  double best_log_prob = 0.0;

  t_total.Start();

  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  for (const auto it : mms_) {
    const auto &classname = it.first;
    const auto &mm = it.second;

    double log_prob = 0.0;
    for (size_t i = 0; i < locs.size(); i++) {
      const auto &loc = locs[i];
      log_prob += mm.GetLogProbability(loc, los[i] > 0);
    }

    if (first || log_prob > best_log_prob) {
      best_log_prob = log_prob;
      best_classname = classname;
    }

    first = false;
  }
  double total_ms = t_total.GetMs();

  // Add to results
  results_mutex_.lock();

  confusion_matrix_[w.classname][best_classname]++;

  total_time_ms_ += total_ms;
  timing_counts_++;

  results_mutex_.unlock();
}

void Evaluator::PrintConfusionMatrix() const {
  results_mutex_.lock();

  printf("Took %5.3f ms / evaluation\n",  total_time_ms_ / timing_counts_);

  printf("%15s  ", "");
  for (const auto it1 : confusion_matrix_) {
    const auto &classname1 = it1.first;
    printf("%15s  ", classname1.c_str());
  }
  printf("\n");

  for (const auto it1 : confusion_matrix_) {
    const auto &classname1 = it1.first;
    printf("%15s  ", classname1.c_str());

    int sum = 0;
    for (const auto it2 : it1.second) {
      sum += it2.second;
    }

    for (const auto it2 : it1.second) {
      printf("%15.1f %%", sum == 0 ? 0:(100.0 * it2.second/sum));
    }
    printf("\t(%05d Evaluations)\n", sum);
  }

  results_mutex_.unlock();
}

std::vector<std::string> Evaluator::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it : mms_) {
    classes.push_back(it.first);
  }

  return classes;
}

} // namespace kitti_occ_grids
} // namespace app

