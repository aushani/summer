#pragma once

#include <map>

#include "library/geometry/point.h"

#include "ray_model.h"

namespace ge = library::geometry;

struct ObjectState {
  ge::Point pos = ge::Point(0, 0);
  double angle = 0;

  bool operator<(const ObjectState &os) const {
    if (pos.x != os.pos.x)
      return pos.x < os.pos.x;
    return pos.y < os.pos.y;
  }
};

class DetectionMap {
 public:
  DetectionMap(double size, double res, const RayModel &model);

  void ProcessObservations(const std::vector<ge::Point> &hits);

  double Lookup(const ge::Point &p);

  const std::map<ObjectState, double>& GetScores() const;

 private:
  std::map<ObjectState, double> scores_;

  RayModel model_;

};
