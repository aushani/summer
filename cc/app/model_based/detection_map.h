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
  std::string classname;

  ObjectState(double x, double y, double a, const std::string &cn) :
    pos(x, y), angle(a), classname(cn) { }

  bool operator<(const ObjectState &os) const {
    if (std::abs(pos.x - os.pos.x) > 0.001)
      return pos.x < os.pos.x;

    if (std::abs(pos.y - os.pos.y) > 0.001)
      return pos.y < os.pos.y;

    if (std::abs(angle - os.angle) > 0.001)
      return angle < os.angle;

    return classname < os.classname;
  }
};

class DetectionMap {
 public:
  DetectionMap(double size, double res, const ModelBank &model_bank);

  std::vector<std::string> GetClasses() const;

  void ProcessObservations(const std::vector<ge::Point> &hits);

  double EvaluateObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state) const;

  std::map<ObjectState, double> GetMaxDetections(double thresh_score);

  const std::map<ObjectState, double>& GetScores() const;

  double GetProb(const ObjectState &os) const;
  double GetLogOdds(const ObjectState &os) const;
  double GetScore(const ObjectState &os) const;

 private:
  double size_;
  double res_;
  double angle_res_ = (2.0*M_PI)/1.0;
  std::map<ObjectState, double> scores_;

  ModelBank model_bank_;

  void ProcessObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state);
  void ProcessObservationsWorker(const std::vector<Eigen::Vector2d> &x_hits, std::deque<ObjectState> *states, std::mutex *mutex);

  void GetMaxDetectionsWorker(std::deque<ObjectState> *states, std::map<ObjectState, double> *result, std::mutex *mutex);
  bool IsMaxDetection(const ObjectState &state);
};
