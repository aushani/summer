#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/geometry/point.h"

#include "ray_model.h"

namespace ge = library::geometry;

struct ObjectState {
  ge::Point pos = ge::Point(0, 0);
  double angle = 0;

  ObjectState(double x, double y, double a) :
    pos(x, y), angle(a) { }

  bool operator<(const ObjectState &os) const {
    if (pos.x != os.pos.x)
      return pos.x < os.pos.x;
    if (pos.y != os.pos.y)
      return pos.y < os.pos.y;
    return angle < os.angle;
  }
};

class DetectionMap {
 public:
  DetectionMap(double size, double res, const RayModel &model);

  void ProcessObservations(const std::vector<ge::Point> &hits);

  void ListMaxDetections();

  double Lookup(const ge::Point &p, double angle);

  const std::map<ObjectState, double>& GetScores() const;

 private:
  double size_;
  double res_;
  std::map<ObjectState, double> scores_;

  RayModel model_;

  void ProcessObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state);
  void ProcessObservationsWorker(const std::vector<Eigen::Vector2d> &x_hits, std::deque<ObjectState> *states, std::mutex *mutex);
};
