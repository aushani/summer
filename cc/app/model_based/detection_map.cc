#include "detection_map.h"

#include <Eigen/Core>

DetectionMap::DetectionMap(double size, double res, const RayModel &model) :
 model_(model) {
  for (double x = -size; x <= size; x += res) {
    for (double y = -size; y <= size; y += res) {
      ObjectState s;
      s.pos = ge::Point(x, y);
      scores_[s] = 0.0;
    }
  }
}

void DetectionMap::ProcessObservation(const ge::Point &hit) {
  Eigen::Vector2d x_hit;
  x_hit(0) = hit.x;
  x_hit(1) = hit.y;

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    Eigen::Vector2d x_sensor_object;
    x_sensor_object(0) = it->first.pos.x;
    x_sensor_object(1) = it->first.pos.y;
    double object_angle = it->first.angle;
    double update = model_.EvaluateObservation(x_sensor_object, object_angle, x_hit);

    it->second += update;
  }
}

void DetectionMap::ProcessObservations(const std::vector<ge::Point> &hits) {
  for (auto h : hits) {
    ProcessObservation(h);
  }
  printf("Done\n");
}

double DetectionMap::Lookup(const ge::Point &p) {
  ObjectState s;
  s.pos = p;

  return scores_[s];
}

const std::map<ObjectState, double>& DetectionMap::GetScores() const {
  return scores_;
}
