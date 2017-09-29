#pragma once

#include <vector>
#include <string>

#include <boost/serialization/map.hpp>

#include "library/ray_tracing/feature_occ_grid.h"

#include "library/feature/feature_model.h"
#include "library/feature/model_key.h"

namespace library {
namespace feature {

class ModelBank {
 public:
  ModelBank(double range_xy, double range_z, double res);

  void AddClass(const std::string &classname, int angle_bins);

  std::vector<std::string> GetClasses() const;
  const std::map<ModelKey, FeatureModel>& GetModels() const;

  int GetNumAngleBins(const std::string &classname) const;
  FeatureModel& GetFeatureModel(const std::string &classname, int angle_bin);

  void Save(const char *fn) const;
  static ModelBank Load(const char *fn);

 private:
  float resolution_;
  float range_xy_;
  float range_z_;

  std::map<ModelKey, FeatureModel> models_;
  std::map<std::string, int> angle_bins_;

  // Just for easier boost serialization
  ModelBank() {};

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & resolution_;
    ar & range_xy_;
    ar & range_z_;

    ar & models_;
    ar & angle_bins_;
  }
};

} // namespace chow_liu_tree
} // namespace library
