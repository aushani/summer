#pragma once

#include <memory>

#include "library/chow_liu_tree/marginal_model.h"
#include "library/ray_tracing/device_occ_grid.h"

#include "library/detector/detection_map.h"
#include "library/detector/object_state.h"

namespace clt = library::chow_liu_tree;
namespace rt = library::ray_tracing;

namespace library {
namespace detector {

// Forward declarations
typedef struct DeviceData DeviceData;

class Detector {
 public:
  Detector(double res, double range_x, double range_y);
  ~Detector();

  void AddModel(const std::string &classname, const clt::MarginalModel &mm);

  DetectionMap Run(const rt::DeviceOccGrid &dog);

 private:
  static constexpr int kThreadsPerBlock_ = 1024;

  const double range_x_;
  const double range_y_;
  const int n_x_;
  const int n_y_;

  const double res_;

  std::unique_ptr<DeviceData> device_data_;
  std::vector<std::string> class_names_;
};

} // namespace detector
} // namespace library
