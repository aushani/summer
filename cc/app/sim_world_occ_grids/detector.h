#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/util/angle.h"
#include "library/ray_tracing/dense_occ_grid.h"

#include "app/sim_world_occ_grids/model.h"

namespace rt = library::ray_tracing;
namespace ut = library::util;

namespace app {
namespace sim_world_occ_grids {

struct ObjectState {
  double x;
  double y;

  double theta;

  ObjectState(double xx, double yy, double tt) :
    x(xx), y(yy), theta(ut::MinimizeAngle(tt)) {}

  bool operator<(const ObjectState &rhs) const {
    if (std::abs(x - rhs.x) >= kTolerance_) {
      return x < rhs.x;
    }

    if (std::abs(y - rhs.y) >= kTolerance_) {
      return y < rhs.y;
    }

    if (std::abs(theta - rhs.theta) >= kTolerance_) {
      return theta < rhs.theta;
    }

    return false;
  }

 private:
  static constexpr double kTolerance_ = 0.001;

};

class Detector {
 public:
  Detector(double res, double range_x, double range_y);

  void Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model);

  void Evaluate(const rt::OccGrid &scene, const Model &model, const Model &bg_model);

  //const std::map<ObjectState, double>& GetScores() const;
  double GetScore(const ObjectState &os) const;

  double GetRangeX() const;
  double GetRangeY() const;
  bool InRange(const ObjectState &os) const;

  size_t GetIndex(const ObjectState &os) const;
  ObjectState GetState(size_t idx) const;

  double GetRes() const;

 private:
  double range_x_;
  double range_y_;
  size_t n_x_;
  size_t n_y_;

  double res_;

  //std::map<ObjectState, double> scores_;
  std::vector<double> scores_;

  double Evaluate(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, const ObjectState &state);
  void EvaluateWorkerThread(const rt::DenseOccGrid &scene, const Model &model, const Model &bg_model, std::deque<size_t> *work_queue, std::mutex *mutex);
};

} // namespace sim_world_occ_grids
} // namespace app
