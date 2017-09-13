#pragma once

#include <memory>
#include <Eigen/Core>

#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/occ_grid_builder.h"

#include "library/detector/detection_map.h"
#include "library/detector/object_state.h"
#include "library/detector/device_scores.h"

namespace clt = library::chow_liu_tree;
namespace rt = library::ray_tracing;

namespace library {
namespace detector {

// Forward declarations
typedef struct DeviceData DeviceData;

struct ModelKey {
  std::string classname;
  int angle_bin;

  ModelKey(const std::string &cn, int b) :
    classname(cn), angle_bin(b) {}

  bool operator<(const ModelKey &k) const {
    if (classname != k.classname) {
      return classname < k.classname;
    }

    return angle_bin < k.angle_bin;
  }
};

class Detector {
 public:
  Detector(float res, float range_x, float range_y);
  ~Detector();

  void AddModel(const std::string &classname, int angle_bin, const clt::JointModel &mm, float log_prior=0.0);
  void UpdateModel(const std::string &classname, int angle_bin, const clt::JointModel &jm);
  void UpdateModel(const std::string &classname, int angle_bin, const rt::DeviceOccGrid &dog);

  void LoadIntoJointModel(const std::string &classname, int angle_bin, clt::JointModel *jm) const;

  void Run(const std::vector<Eigen::Vector3d> &hits);

  const DeviceScores& GetScores(const std::string &classname, int angle_bin) const;

  float GetScore(const std::string &classname, const ObjectState &os) const;
  float GetProb(const std::string &classname, const ObjectState &os) const;
  float GetLogOdds(const std::string &classname, const ObjectState &os) const;

  float GetProb(const std::string &classname, double x, double y) const;
  float GetLogOdds(const std::string &classname, double x, double y) const;

  float GetResolution() const;
  float GetRangeX() const;
  float GetRangeY() const;
  bool InRange(const ObjectState &os) const;

  float GetNX() const;
  float GetNY() const;

  const std::vector<std::string>& GetClasses() const;

 private:
  static constexpr int kThreadsPerBlock_ = 128;

  const float range_x_;
  const float range_y_;
  const int n_x_;
  const int n_y_;

  const float res_;

  std::unique_ptr<DeviceData> device_data_;
  std::vector<std::string> classnames_;
  rt::OccGridBuilder og_builder_;

  int GetModelIndex(const std::string &classname) const;
};

} // namespace detector
} // namespace library
