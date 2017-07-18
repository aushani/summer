#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/geometry/point.h"

#include "ray_model.h"
#include "model_bank.h"

namespace ge = library::geometry;

struct ObjectState {
  ge::Point pos = ge::Point(0, 0);
  double angle = 0;

  ObjectState(double x, double y, double a) :
    pos(x, y), angle(a) { }

  bool operator<(const ObjectState &os) const {
    if (std::abs(pos.x - os.pos.x) > 0.001)
      return pos.x < os.pos.x;

    if (std::abs(pos.y - os.pos.y) > 0.001)
      return pos.y < os.pos.y;

    if (std::abs(angle - os.angle) > 0.001)
      return angle < os.angle;

    return false;
  }
};

class DetectionMap {
 public:
  DetectionMap(double size, double res, const ModelBank &model_bank);

  void ProcessObservations(const std::vector<ge::Point> &hits);

  std::map<ObjectState, double> GetMaxDetections(double thresh_score);

  double Lookup(const ge::Point &p, double angle);

  const std::map<ObjectState, double>& GetScores() const;

 private:
  double size_;
  double res_;
  std::map<ObjectState, double> scores_;

  ModelBank model_bank_;

  void ProcessObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state);
  void ProcessObservationsWorker(const std::vector<Eigen::Vector2d> &x_hits, std::deque<ObjectState> *states, std::mutex *mutex);

  void GetMaxDetectionsWorker(std::deque<ObjectState> *states, std::map<ObjectState, double> *result, std::mutex *mutex);
  bool IsMaxDetection(const ObjectState &state);
};
