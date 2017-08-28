#include "app/sim_world_occ_grids/detector.h"

#include <stdlib.h>
#include <iostream>
#include <thread>

#include <boost/assert.hpp>

namespace app {
namespace sim_world_occ_grids {

Detector::Detector(double res, double range_x, double range_y) :
 range_x_(range_x), range_y_(range_y),
  n_x_(2 * std::ceil(range_x / res) + 1),
  n_y_(2 * std::ceil(range_y / res) + 1),
  res_(res),
  scores_(n_x_*n_y_, 0.0) {
}

void Detector::Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model) {
  BOOST_ASSERT(scene.GetResolution() == model.GetResolution());

  std::deque<size_t> work_queue;

  for (size_t i = 0; i< scores_.size(); i++) {
    work_queue.push_back(i);
  }

  int num_threads = 48;
  printf("Have %ld states to evaluate with %d threads\n",
      work_queue.size(), num_threads);

  std::vector<std::thread> threads;
  std::mutex mutex;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&Detector::EvaluateWorkerThread, this, scene, model, bg_model, &work_queue, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void Detector::Evaluate(const rt::OccGrid &scene, const Model &model, const Model &bg_model) {
  BOOST_ASSERT(scene.GetResolution() == model.GetResolution());

  auto &locs = scene.GetLocations();
  auto &los = scene.GetLogOdds();

  printf("Have %ld voxels to evaluate\n", locs.size());

  // TODO
  double model_size = 5.0;
  double res = scene.GetResolution();

  for (size_t i = 0; i < locs.size(); i++) {
    if (i % 1000 == 0) {
      printf("at %ld / %ld\n", i, locs.size());
    }

    auto &loc = locs[i];
    auto &lo = los[i];

    double x = loc.i * res;
    double y = loc.j * res;

    for (double dix = -model_size/res; dix < model_size/res; dix++) {
      for (double diy = -model_size/res; diy < model_size/res; diy++) {
        double x_at = x + dix * res;
        double y_at = y + diy * res;

        rt::Location loc_at(dix, diy, loc.k);

        double model_score = model.GetProbability(loc_at, lo);
        double bg_score = bg_model.GetProbability(loc_at, lo);

        size_t idx = GetIndex(ObjectState(x_at, y_at, 0));
        scores_[idx] += log(model_score) - log(bg_score);
      }
    }
  }
}

void Detector::EvaluateWorkerThread(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, std::deque<size_t> *work_queue, std::mutex *mutex) {
  bool done = false;

  while (!done) {
    mutex->lock();
    if (work_queue->empty()) {
      done = true;
      mutex->unlock();
    } else {
      size_t idx = work_queue->front();
      work_queue->pop_front();
      mutex->unlock();

      ObjectState os = GetState(idx);
      double score = Evaluate(scene, model, bg_model, os);

      scores_[idx] = 1 / (1 + exp(-score));
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

double Detector::GetScore(const ObjectState &os) const {
  if (!InRange(os)) {
    return 0.0;
  }

  size_t idx = GetIndex(os);
  return scores_[idx];
}

double Detector::GetRangeX() const {
  return range_x_;
}

double Detector::GetRangeY() const {
  return range_y_;
}

bool Detector::InRange(const ObjectState &os) const {
  return std::abs(os.x) < range_x_ && std::abs(os.y) < range_y_;
}

size_t Detector::GetIndex(const ObjectState &os) const {
  int ix = os.x / res_ + n_x_ / 2;
  int iy = os.y / res_ + n_y_ / 2;

  if (ix >= n_x_ || iy >= n_y_) {
    return -1;
  }

  size_t idx = ix * n_y_ + iy;
  return idx;
}

ObjectState Detector::GetState(size_t idx) const {
  size_t ix = idx / n_y_;
  size_t iy = idx % n_y_;

  // int instead of size_t because could be negative
  int dix = ix - n_x_/2;
  int diy = iy - n_y_/2;

  double x = dix * res_;
  double y = diy * res_;

  return ObjectState(x, y, 0);
}

double Detector::GetRes() const {
  return res_;
}

} // namespace sim_world_occ_grids
} // namespace app
