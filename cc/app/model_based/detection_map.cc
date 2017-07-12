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

void DetectionMap::ProcessObservations(const std::vector<ge::Point> &hits) {
  std::vector<Eigen::Vector2d> x_hits;
  for (auto h : hits) {
    Eigen::Vector2d hit;
    hit << h.x, h.y;
    x_hits.push_back(hit);
  }

  for (auto it = scores_.begin(); it != scores_.end(); it++) {
    Eigen::Vector2d x_sensor_object;
    x_sensor_object(0) = it->first.pos.x;
    x_sensor_object(1) = it->first.pos.y;
    double object_angle = it->first.angle;
    double update = model_.EvaluateObservations(x_sensor_object, object_angle, x_hits);

    it->second += update;
  }
  printf("Done\n");
}

double DetectionMap::Lookup(const ge::Point &p) {
  ObjectState s;
  s.pos = p;

  double score = scores_[s];

  if (score < -100)
    return 0.0f;
  if (score > 100)
    return 1.0f;
  return 1/(1+exp(-score));
}

const std::map<ObjectState, double>& DetectionMap::GetScores() const {
  return scores_;
}
