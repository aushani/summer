#pragma once

#include <vector>

#include <boost/serialization/vector.hpp>

#include "library/ray_tracing/feature_occ_grid.h"

#include "library/feature/counter.h"

namespace rt = library::ray_tracing;

namespace library {
namespace feature {

class FeatureModel {
 public:
  FeatureModel(double range_xy, double range_z, double res);

  void MarkObservations(const rt::FeatureOccGrid &fog);

  const Counter& GetCounter(const rt::Location &loc) const;
  int GetCount(const rt::Location &loc, bool occ) const;
  int GetCount(const rt::Location &loc, float theta, float phi) const;
  int GetNumOccuObservations(const rt::Location &loc) const;
  int GetNumFeatureObservations(const rt::Location &loc) const;

  int GetMode(const rt::Location &loc, float *theta, float *phi) const;

  double GetResolution() const;
  size_t GetNXY() const;
  size_t GetNZ() const;
  bool InRange(const rt::Location &loc) const;

  void Save(const char *fn) const;
  static FeatureModel Load(const char *fn);

  float kAngleRes = 0.2; // About 10 degrees, ~1000 bins
 private:

  float resolution_;
  float range_xy_;
  float range_z_;

  size_t n_xy_;
  size_t n_z_;
  size_t n_loc_;

  std::vector<Counter> counters_;

  // for easier boost serialization
  FeatureModel();

  void MarkOccuWorker(const rt::FeatureOccGrid &fog, size_t idx_start, size_t idx_end);
  void MarkFeaturesWorker(const rt::FeatureOccGrid &fog, size_t idx_start, size_t idx_end);

  size_t GetIndex(const rt::Location &loc) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & kAngleRes;

    ar & resolution_;
    ar & range_xy_;
    ar & range_z_;

    ar & n_xy_;
    ar & n_z_;

    ar & counters_;
  }
};

} // namespace chow_liu_tree
} // namespace library
