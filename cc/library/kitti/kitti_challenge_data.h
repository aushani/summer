#pragma once

#include <Eigen/Core>

#include "library/kitti/velodyne_scan.h"
#include "library/kitti/object_label.h"

namespace library {
namespace kitti {

class KittiChallengeData {
 public:
  static KittiChallengeData LoadFrame(const std::string &dirname, int frame);

  const VelodyneScan& GetScan() const;
  const ObjectLabels& GetLabels() const;
  const Eigen::Matrix4d& GetTcv() const;

 private:
  VelodyneScan scan_;
  ObjectLabels labels_;
  Eigen::Matrix4d t_cv_;

  KittiChallengeData(const VelodyneScan &scan, const ObjectLabels &labels, const Eigen::Matrix4d &t);

  static VelodyneScan LoadVelodyneScan(const std::string &dirname, int frame_num);
  static ObjectLabels LoadObjectLabels(const std::string &dirname, int frame_num);
  static Eigen::Matrix4d LoadCalib(const std::string &dirname, int frame_num);
};

} // namespace kitti
} // namespace library
