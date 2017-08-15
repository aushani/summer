#pragma once

#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "library/kitti/velodyne_scan.h"

#include "app/kitti/ray_model.h"
#include "app/kitti/model_bank.h"
#include "app/kitti/observation.h"
#include "app/kitti/object_state.h"

namespace kt = library::kitti;

namespace app {
namespace kitti {

class DetectionMap {
 public:
  DetectionMap(double size_xy, const ModelBank &model_bank);

  std::vector<std::string> GetClasses() const;

  void ProcessScan(const kt::VelodyneScan &scan);

  double EvaluateScanForState(const kt::VelodyneScan &scan, const ObjectState &state) const;

  const std::map<ObjectState, double>& GetScores() const;

  double GetProb(const ObjectState &os) const;
  double GetLogOdds(const ObjectState &os) const;
  double GetScore(const ObjectState &os) const;

 private:
  const double kPosRes_ = 0.5;                  // 50 cm
  const double kAngleRes_ = 30.0 * M_PI/180.0;  // 30 deg

  double size_xy_;

  std::map<ObjectState, double> scores_;

  ModelBank model_bank_;

  void ProcessObservationsWorker(const std::vector<Observation> &obs, std::deque<std::vector<ObjectState> > *states, std::mutex *mutex);
};

} // namespace kitti
} // namespace app
