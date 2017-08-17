#include "app/kitti_occ_grids/detector.h"

#include <stdlib.h>
#include <iostream>
#include <thread>

#include <boost/assert.hpp>

namespace app {
namespace kitti_occ_grids {

Detector::Detector(double res, int n_size) {
  for (int ix = -n_size; ix < n_size; ix++) {
    double x = ix * res;
    for (int iy = -n_size; iy < n_size; iy++) {
      double y = iy * res;
      scores_[ObjectState(x, y)] = 0;
    }
  }
}

void Detector::Evaluate(const rt::DenseOccGrid &scene, const rt::OccGrid &model) {
  BOOST_ASSERT(scene.GetResolution() == model.GetResolution());

  std::deque<ObjectState> states;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    states.push_back(it->first);
  }

  int num_threads = 48;

  std::vector<std::thread> threads;
  std::mutex mutex;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&Detector::EvaluateWorkerThread, this, scene, model, &states, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }

}

void Detector::EvaluateWorkerThread(const rt::DenseOccGrid &scene, const rt::OccGrid &model, std::deque<ObjectState> *states, std::mutex *mutex) {
  bool done = false;

  while (!done) {
    mutex->lock();
    if (states->empty()) {
      done = true;
      mutex->unlock();
    } else {
      auto s = states->front();
      states->pop_front();
      mutex->unlock();

      scores_[s] = Evaluate(scene, model, s);
    }
  }
}

double Detector::Evaluate(const rt::DenseOccGrid &scene, const rt::OccGrid &model, const ObjectState &state) {
  auto model_locs = model.GetLocations();
  auto model_los = model.GetLogOdds();

  float res = model.GetResolution();

  double log_score = 0.0;

  for (size_t i = 0; i < model_locs.size(); i++) {
    auto loc = model_locs[i];
    float lo = model_los[i];
    double p_model = 1 / (1 + exp(-lo));

    double dx = loc.i * res;
    double dy = loc.j * res;
    double dz = loc.k * res;

    rt::Location scene_loc(state.x + dx, state.y + dy, dz, res);

    double p_scene = scene.GetProbability(scene_loc);

    double score = (p_model * p_scene) + (1-p_model) * (1-p_scene);

    log_score += log(score);
  }

  return log_score;
}

const std::map<ObjectState, double>& Detector::GetScores() const {
  return scores_;
}

} // namespace kitti_occ_grids
} // namespace app
