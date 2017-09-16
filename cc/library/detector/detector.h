#pragma once

#include <memory>
#include <Eigen/Core>

#include "library/feature/feature_model.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/occ_grid_builder.h"

#include "library/detector/detection_map.h"
#include "library/detector/object_state.h"
#include "library/detector/device_scores.h"
#include "library/detector/dim.h"

namespace ft = library::feature;
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

  bool operator==(const ModelKey &k) const {
    return classname == k.classname && angle_bin == k.angle_bin;
  }

  bool operator!=(const ModelKey &k) const {
    return !( (*this)==k );
  }
};

struct Detection {
  std::string classname;
  ObjectState os;

  float confidence;

  Detection(const std::string &cn, const ObjectState &o, float c) :
    classname(cn), os(o), confidence(c) {}

  bool operator<(const Detection &d) const {
    return confidence > d.confidence;
  }
};

class Detector {
 public:
  Detector(const Dim &d);
  ~Detector();

  void AddModel(const std::string &classname, int angle_bin, const ft::FeatureModel &fm, float log_prior=0.0);

  void Run(const std::vector<Eigen::Vector3d> &hits, const std::vector<float> &intensities);

  std::vector<Detection> GetDetections(double thresh) const;

  const DeviceScores& GetScores(const std::string &classname, int angle_bin) const;

  float GetScore(const std::string &classname, const ObjectState &os) const;
  float GetProb(const std::string &classname, const ObjectState &os) const;
  float GetLogOdds(const std::string &classname, const ObjectState &os) const;

  float GetProb(const std::string &classname, double x, double y) const;
  float GetLogOdds(const std::string &classname, double x, double y) const;

  const Dim& GetDim() const;

  const std::vector<std::string>& GetClasses() const;

  static constexpr int kAngleBins = 1;
  bool use_features = true;
 private:
  static constexpr int kThreadsPerBlock_ = 128;

  const Dim dim_;

  std::unique_ptr<DeviceData> device_data_;
  std::vector<std::string> classnames_;
  rt::OccGridBuilder og_builder_;

  int GetModelIndex(const std::string &classname) const;
};

} // namespace detector
} // namespace library
