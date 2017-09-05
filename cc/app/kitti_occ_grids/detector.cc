#include "app/kitti_occ_grids/detector.h"

#include <stdlib.h>
#include <iostream>
#include <thread>

#include <boost/assert.hpp>

namespace app {
namespace kitti_occ_grids {

Detector::Detector(double res, double range_x, double range_y) :
 range_x_(range_x), range_y_(range_y),
  n_x_(2 * std::ceil(range_x / res) + 1),
  n_y_(2 * std::ceil(range_y / res) + 1),
  res_(res) {
}

void Detector::AddModel(const std::string &classname, const clt::MarginalModel &mm) {
  class_scores_.insert({classname, std::vector<double>(n_x_*n_y_, 0.0)});
  models_.insert({classname, mm});
}

void Detector::Evaluate(const rt::DenseOccGrid &scene) {
  std::deque<size_t> work_queue;

  for (size_t i = 0; i < n_x_ * n_y_; i++) {
    work_queue.push_back(i);
  }

  int num_threads = 48;
  printf("Have %ld states to evaluate with %d threads\n",
      work_queue.size(), num_threads);

  std::vector<std::thread> threads;
  std::mutex mutex;
  for (int i=0; i<num_threads; i++) {
    threads.push_back(std::thread(&Detector::EvaluateWorkerThread, this, scene, &work_queue, &mutex));
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void Detector::EvaluateWorkerThread(const rt::DenseOccGrid &scene, std::deque<size_t> *work_queue, std::mutex *mutex) {
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

      Evaluate(scene, idx);
    }
  }
}

void Detector::Evaluate(const rt::DenseOccGrid &scene, size_t idx) {
  ObjectState os = GetState(idx);

  // Evaluate all locations for all models
  for (const auto &kv : models_) {
    const auto &classname = kv.first;
    const auto &mm = kv.second;

    const auto &it_scores = class_scores_.find(classname);
    BOOST_ASSERT(it_scores != class_scores_.end());
    auto &scores = it_scores->second;

    int min_ij = - (mm.GetNXY() / 2);
    int max_ij = min_ij + mm.GetNXY();

    int min_k = - (mm.GetNZ() / 2);
    int max_k = min_k + mm.GetNZ();

    for (int i=min_ij; i < max_ij; i++) {
      for (int j=min_ij; j < max_ij; j++) {
        for (int k=min_k; k < max_k; k++) {
          rt::Location loc(i, j, k);
          rt::Location loc_global(i + os.x/res_, j + os.y/res_, k);

          if (!scene.IsKnown(loc_global)) {
            continue;
          }

          bool occ = scene.GetProbability(loc_global) > 0.5;

          int c = mm.GetCount(loc, occ);
          double denom = mm.GetNumObservations(loc);

          double p = c / denom;
          scores[idx] += log(p);
        }
      }
    }
  }
}

double Detector::GetScore(const std::string &classname, const ObjectState &os) const {
  if (!InRange(os)) {
    return 0.0;
  }

  const auto &it = class_scores_.find(classname);
  BOOST_ASSERT(it != class_scores_.end());
  const auto &scores = it->second;

  size_t idx = GetIndex(os);
  return scores[idx];
}

double Detector::GetProb(const std::string &classname, const ObjectState &os) const {
  if (!InRange(os)) {
    return 0.0;
  }

  size_t idx = GetIndex(os);

  const auto &it = class_scores_.find(classname);
  BOOST_ASSERT(it != class_scores_.end());
  const auto &scores = it->second;
  double my_score = scores[idx];

  double max_score = my_score;

  for (const auto &kv : class_scores_) {
    const auto &classname = kv.first;

    const auto &it = class_scores_.find(classname);
    BOOST_ASSERT(it != class_scores_.end());
    const auto &scores = it->second;
    double score = scores[idx];

    if (score > max_score) {
      max_score = score;
    }
  }

  double sum = 0;

  for (const auto &kv : class_scores_) {
    const auto &classname = kv.first;

    const auto &it = class_scores_.find(classname);
    BOOST_ASSERT(it != class_scores_.end());
    const auto &scores = it->second;
    double score = scores[idx];

    sum += exp(score - max_score);
  }

  double prob = exp(my_score - max_score) / sum;
  double lo = -log(1/prob - 1);
  if (lo > 10) {
    lo = 10;
  }

  if (lo < -10) {
    lo = -10;
  }

  return lo;
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

} // namespace kitti_occ_grids
} // namespace app
