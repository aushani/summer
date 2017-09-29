#include "library/feature/model_bank.h"

#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace library {
namespace feature {

ModelBank::ModelBank(double range_xy, double range_z, double res) :
 resolution_(res),
 range_xy_(range_xy),
 range_z_(range_z) {
}

void ModelBank::AddClass(const std::string &classname, int angle_bins) {
  angle_bins_[classname] = angle_bins;

  for (int i=0; i < angle_bins; i++) {
    ModelKey key(classname, i);
    models_.insert({key, FeatureModel(range_xy_, range_z_, resolution_)});
  }
}

std::vector<std::string> ModelBank::GetClasses() const {
  std::vector<std::string> classes;
  for (const auto &kv : angle_bins_) {
    classes.push_back(kv.first);
  }

  return classes;
}

const std::map<ModelKey, FeatureModel>& ModelBank::GetModels() const {
  return models_;
}

int ModelBank::GetNumAngleBins(const std::string &classname) const {
  auto it = angle_bins_.find(classname);
  if (it == angle_bins_.end()) {
    return 0;
  }

  return it->second;
}

FeatureModel& ModelBank::GetFeatureModel(const std::string &classname, int angle_bin) {
  auto it = models_.find(ModelKey(classname, angle_bin));
  BOOST_ASSERT(it != models_.end());

  return it->second;
}

void ModelBank::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

ModelBank ModelBank::Load(const char *fn) {
  ModelBank mb;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> mb;

  return mb;
}

} // namespace chow_liu_tree
} // namespace library
