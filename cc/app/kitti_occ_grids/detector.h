#pragma once

#include <map>
#include <deque>
#include <mutex>

#include "library/util/angle.h"
#include "library/ray_tracing/dense_occ_grid.h"

#include "library/chow_liu_tree/marginal_model.h"
#include "library/detector/detector.h"

namespace clt = library::chow_liu_tree;
namespace dt = library::detector;
namespace rt = library::ray_tracing;
namespace ut = library::util;

namespace app {
namespace kitti_occ_grids {

class Detector {
 public:
  Detector(double res, double range_x, double range_y);

  void AddModel(const std::string &classname, const clt::MarginalModel &mm);

  void Evaluate(const rt::DeviceOccGrid &scene);

  double GetScore(const std::string &classname, const ObjectState &os) const;
  double GetProb(const std::string &classname, const ObjectState &os) const;

  double GetRangeX() const;
  double GetRangeY() const;
  bool InRange(const ObjectState &os) const;

  double GetRes() const;

 private:
  double range_x_;
  double range_y_;
  size_t n_x_;
  size_t n_y_;

  double res_;

  std::map<std::string, std::vector<double> > classnames_;

  dt::Detector detector_;
};

} // namespace kitti_occ_grids
} // namespace app
