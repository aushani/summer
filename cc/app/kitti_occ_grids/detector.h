#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/ray_tracing/dense_occ_grid.h"

#include "app/kitti_occ_grids/model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

struct ObjectState {
  double x;
  double y;

  ObjectState(double xx, double yy) :
    x(xx), y(yy) {}

  bool operator<(const ObjectState &rhs) const {
    if (x != rhs.x) {
      return x < rhs.x;
    }

    return y < rhs.y;
  }
};

class Detector {
 public:
  Detector(double res, int n_size);

  void Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model);

  const std::map<ObjectState, double>& GetScores() const;

 private:
  std::map<ObjectState, double> scores_;

  double Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, const ObjectState &state);
  void EvaluateWorkerThread(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, std::deque<ObjectState> *states, std::mutex *mutex);

};

} // namespace kitti_occ_grids
} // namespace app
