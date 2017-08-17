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

void Detector::Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model) {
  BOOST_ASSERT(scene.GetResolution() == model.GetResolution());

  std::deque<ObjectState> states;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    states.push_back(it->first);
  }

  int num_threads = 48;

  std::vector<std::thread> threads;
  std::mutex mutex;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&Detector::EvaluateWorkerThread, this, scene, model, bg_model, &states, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }

}

void Detector::EvaluateWorkerThread(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, std::deque<ObjectState> *states, std::mutex *mutex) {
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

      double score = Evaluate(scene, model, bg_model, s);

      scores_[s] = 1 / (1 + exp(-score));
    }
  }
}

double Detector::Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, const ObjectState &state) {
  auto counts = model.GetCounts();
  auto bg_counts = bg_model.GetCounts();

  float res = model.GetResolution();

  double log_score = 0.0;

  for (auto it = counts.cbegin(); it != counts.cend(); it++) {
    // Check to make sure it exists in background
    auto it_bg = bg_counts.find(it->first);
    if (it_bg == bg_counts.end()) {
      continue;
    }

    double dx = it->first.i * res;
    double dy = it->first.j * res;
    double dz = it->first.k * res;
    rt::Location scene_loc(state.x + dx, state.y + dy, dz, res);

    float lo_scene = scene.GetLogOdds(scene_loc);

    // Check if this is unknown
    if (std::abs(lo_scene) < 1e-3) {
      continue;
    }

    double model_score = it->second.GetProbability(lo_scene);
    double bg_score = it_bg->second.GetProbability(lo_scene);

    if (model_score < 1e-3) {
      model_score = 1e-3;
    }

    if (bg_score < 1e-3) {
      bg_score = 1e-3;
    }

    log_score += log(model_score) - log(bg_score);
  }

  return log_score;
}

const std::map<ObjectState, double>& Detector::GetScores() const {
  return scores_;
}

} // namespace kitti_occ_grids
} // namespace app
