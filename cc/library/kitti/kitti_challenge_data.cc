#include "library/kitti/kitti_challenge_data.h"

namespace library {
namespace kitti {

KittiChallengeData::KittiChallengeData(const VelodyneScan &scan, const ObjectLabels &labels, const Eigen::Matrix4d &t) :
 scan_(scan), labels_(labels), t_cv_(t) {}

KittiChallengeData KittiChallengeData::LoadFrame(const std::string &dirname, int frame) {
  VelodyneScan vs = LoadVelodyneScan(dirname, frame);
  ObjectLabels ols = LoadObjectLabels(dirname, frame);
  Eigen::Matrix4d t = LoadCalib(dirname, frame);

  return KittiChallengeData(vs, ols, t);
}

VelodyneScan KittiChallengeData::LoadVelodyneScan(const std::string &dirname, int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/data_object_velodyne/training/velodyne/%06d.bin",
      dirname.c_str(), frame_num);

  printf("Loading velodyne from %s\n", fn);

  return VelodyneScan(fn);
}

ObjectLabels KittiChallengeData::LoadObjectLabels(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_label_2/training/label_2/%06d.txt",
      dirname.c_str(), frame_num);

  printf("Loading labels from %s\n", fn);

  ObjectLabels labels = ObjectLabel::Load(fn);

  return labels;
}

Eigen::Matrix4d KittiChallengeData::LoadCalib(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_calib/training/calib/%06d.txt",
      dirname.c_str(), frame_num);

  printf("Loading calib from %s\n", fn);

  Eigen::MatrixXd T_cv = ObjectLabel::LoadVelToCam(fn);

  return T_cv;
}

const VelodyneScan& KittiChallengeData::GetScan() const {
  return scan_;
}

const ObjectLabels& KittiChallengeData::GetLabels() const {
  return labels_;
}

const Eigen::Matrix4d& KittiChallengeData::GetTcv() const {
  return t_cv_;
}


} // namespace kitti
} // namespace library
