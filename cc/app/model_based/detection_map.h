#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/geometry/point.h"

#include "ray_model.h"
#include "model_bank.h"
#include "observation.h"
#include "object_state.h"

namespace ge = library::geometry;

class DetectionMap {
 public:
  DetectionMap(double size, double res, const ModelBank &model_bank);

  std::vector<std::string> GetClasses() const;

  void ProcessObservations(const std::vector<ge::Point> &hits);

  double EvaluateObservationsForState(const std::vector<Eigen::Vector2d> &x_hits, const ObjectState &state) const;
  double EvaluateObservationsForState(const std::vector<Observation> &x_hits, const ObjectState &state) const;

  std::map<ObjectState, double> GetMaxDetections(double log_odds_threshold);

  const std::map<ObjectState, double>& GetScores() const;

  double GetProb(const ObjectState &os) const;
  double GetLogOdds(const ObjectState &os) const;
  double GetScore(const ObjectState &os) const;

 private:
  double size_;
  double res_;
  double angle_res_ = 15.0 * M_PI/180.0; // 15 deg
  std::map<ObjectState, double> scores_;

  ModelBank model_bank_;

  void ProcessObservationsForState(const std::vector<Observation> &x_hits, const ObjectState &state);
  void ProcessObservationsWorker(const std::vector<Observation> &x_hits, std::deque<ObjectState> *states, std::mutex *mutex);

  void GetMaxDetectionsWorker(std::deque<ObjectState> *states, std::map<ObjectState, double> *result, std::mutex *mutex);
  bool IsMaxDetection(const ObjectState &state);
};
