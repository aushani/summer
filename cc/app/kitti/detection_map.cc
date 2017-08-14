#include "app/kitti/detection_map.h"

#include <thread>
#include <queue>

#include <boost/assert.hpp>
#include <Eigen/Core>

#include "library/timer/timer.h"

namespace kt = library::kitti;

namespace app {
namespace kitti {

DetectionMap::DetectionMap(double size_xy, double size_z, const ModelBank &model_bank) :
 size_xy_(size_xy), size_z_(size_z), model_bank_(model_bank) {
  auto classes = GetClasses();

  for (std::string classname : classes) {
    double angle_res = kAngleRes_;
    if (classname == std::string("NOOBJ")) {
      angle_res = 2*M_PI;
    }

    for (double x = 0; x <= size_xy_; x += kPosRes_) {
      for (double y = -size_xy_; y <= size_xy_; y += kPosRes_) {
        for (double z = -2; z <= 0; z += kPosRes_) {
          for (double angle = 0; angle < 2*M_PI; angle += angle_res) {
            ObjectState s(Eigen::Vector3d(x, y, z), angle, classname);
            scores_.insert( std::pair<ObjectState, double>(s, 0) );
          }
        }
      }
    }
  }
}

std::vector<std::string> DetectionMap::GetClasses() const {
  std::vector<std::string> classes;
  auto models = model_bank_.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    classes.push_back(it->first);
  }

  return classes;
}

double DetectionMap::EvaluateScanForState(const kt::VelodyneScan &scan, const ObjectState &state) const {
  return model_bank_.EvaluateScan(state, scan);
}

void DetectionMap::ProcessObservationsWorker(const std::vector<Observation> &obs, std::deque< std::vector<ObjectState> > *states, std::mutex *mutex) {
  bool done = false;

  while (!done) {
    mutex->lock();
    if (states->empty()) {
      done = true;
      mutex->unlock();
    } else {
      std::vector<ObjectState> ss = states->front();
      states->pop_front();

      size_t num_left = states->size();

      mutex->unlock();

      if (num_left % 1000 == 0) {
        printf("%ld left...\n", num_left);
      }

      const ObjectState &os = ss[0];
      auto mos = ModelObservation::MakeModelObservations(os, obs, model_bank_.GetMaxSizeXY(), model_bank_.GetMaxSizeZ());

      for (const auto &os : ss) {
        auto it = scores_.find(os);
        BOOST_ASSERT( it != scores_.end());
        it->second += model_bank_.EvaluateModelObservations(os, mos);
      }
    }
  }
}

void DetectionMap::ProcessScan(const kt::VelodyneScan &scan) {
  std::deque<std::vector<ObjectState> > states;
  std::mutex mutex;

  std::vector<ObjectState> ss;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    auto &os = it->first;
    if (ss.empty() ||
          (os.pos.x() == ss[0].pos.x() &&
           os.pos.y() == ss[0].pos.y() &&
           os.pos.z() == ss[0].pos.z() &&
           os.theta   == ss[0].theta)) {
      ss.push_back(os);
    } else {
      states.push_back(ss);
      ss.clear();
      ss.push_back(os);
    }
  }

  if (!ss.empty()) {
    states.push_back(ss);
  }

  // Make observations
  std::vector<Observation> obs;
  for (const auto hit : scan.GetHits()) {
    obs.emplace_back(hit);
  }

  int num_threads = 48;
  printf("\tMaking %d threads to process %ld states\n", num_threads, states.size());

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&DetectionMap::ProcessObservationsWorker, this, obs, &states, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  printf("Done\n");
}

const std::map<ObjectState, double>& DetectionMap::GetScores() const {
  return scores_;
}

double DetectionMap::GetProb(const ObjectState &os) const {
  auto it_os = scores_.find(os);
  if (it_os == scores_.end()) {
    printf("not found!\n");
    return 0.0f;
  }

  double my_score = it_os->second;
  double denom = 0.0;

  auto it = it_os;
  it++;
  while (it != scores_.end() &&
         std::abs(it->first.pos(0) - it_os->first.pos(0)) < 1e-3 &&
         std::abs(it->first.pos(1) - it_os->first.pos(1)) < 1e-3) {
    double s = it->second - my_score;
    denom += exp(s);
    it++;
  }

  it = it_os;
  while (std::abs(it->first.pos(0) - it_os->first.pos(0)) < 1e-3 &&
         std::abs(it->first.pos(1) - it_os->first.pos(1)) < 1e-3) {
    double s = it->second - my_score;
    denom += exp(s);
    if (it == scores_.begin()) {
      break;
    }
    it--;
  }

  double p = 1.0/denom;
  //printf("my score = %5.3f, denom = %5.3f\n", my_score, denom);
  return p;
}

double DetectionMap::GetLogOdds(const ObjectState &os) const {
  double p = GetProb(os);
  if (p < 1e-99)
    p = 1e-99;
  if (p > (1 - 1e-99))
    p = 1 - 1e-99;
  double lo = -log(1.0/p - 1);
  return lo;
}

double DetectionMap::GetScore(const ObjectState &os) const {
  auto it = scores_.find(os);
  if (it == scores_.end()) {
    printf("not found!\n");
    return 0.0f;
  }
  return it->second;
}


} // namespace kitti
} // namespace app
