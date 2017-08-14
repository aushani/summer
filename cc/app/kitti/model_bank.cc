#include "app/kitti/model_bank.h"

#include <iostream>
#include <fstream>
#include <thread>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "library/timer/timer.h"

namespace kt = library::kitti;

namespace app {
namespace kitti {

ModelBank::ModelBank() {

}

void ModelBank::MarkObservations(const ObjectState &os, const std::vector<Observation> &x_hits) {
  // Convert to ModelObservation's
  std::vector<ModelObservation> mos;
  for (const auto &x_hit : x_hits) {
    mos.emplace_back(os, x_hit);
  }

  std::string cn = os.classname;

  if (obj_models_.count(cn) == 0) {
    RayModel model;
    obj_models_.insert( std::pair<std::string, RayModel>(cn, model) );
  }

  auto it = obj_models_.find(cn);
  it->second.MarkObservations(mos);
}

const RayModel& ModelBank::GetModel(const std::string &name) const {
  // Assume class exists
  return obj_models_.find(name)->second;
}

const std::map<std::string, RayModel>& ModelBank::GetModels() const {
  return obj_models_;
}

std::vector<ModelObservation> ModelBank::GetRelevantModelObservations(const std::vector<ModelObservation> &mos) const {
  std::vector<ModelObservation> relevant_mos;
  for (const auto &mo : mos) {
    bool is_relevant = true;
    for (auto it = obj_models_.begin(); is_relevant && it != obj_models_.end(); it++) {
      if (!it->second.IsRelevant(mo)) {
        is_relevant = false;
      }
    }

    if (is_relevant) {
      relevant_mos.push_back(mo);
    }
  }

  return relevant_mos;
}

double ModelBank::EvaluateScan(const ObjectState &os, const kt::VelodyneScan &scan) const {
  auto it = obj_models_.find(os.classname);
  if (it != obj_models_.end()) {
    auto mos = ModelObservation::MakeModelObservations(os, scan, GetMaxSizeXY(), GetMaxSizeZ());

    return it->second.EvaluateObservations(mos);
  }

  return 0.0f;
}

double ModelBank::EvaluateModelObservations(const ObjectState &os, const std::vector<ModelObservation> &mos) const {
  auto it = obj_models_.find(os.classname);
  if (it != obj_models_.end()) {
    return it->second.EvaluateObservations(mos);
  }

  return 0.0f;
}

void ModelBank::PrintStats() const {
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    printf("Class %s\n", it->first.c_str());
    it->second.PrintStats();
  }
}

std::vector<std::string> ModelBank::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    classes.push_back(it->first);
  }

  return classes;
}

double ModelBank::GetMaxSizeXY() const {
  double res = 0;

  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    if (it->second.GetMaxSizeXY() > res) {
      res = it->second.GetMaxSizeXY();
    }
  }

  return res;
}

double ModelBank::GetMaxSizeZ() const {
  double res = 0;

  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    if (it->second.GetMaxSizeZ() > res) {
      res = it->second.GetMaxSizeZ();
    }
  }

  return res;
}

void ModelBank::BlurClass(const std::string &classname) {
  auto it = obj_models_.find(classname);
  if (it == obj_models_.end()) {
    printf("Class %s not found!\n", classname.c_str());
  }

  printf("Blurring class %s\n", it->first.c_str());
  library::timer::Timer t;
  it->second.Blur();
  printf("Took %5.3f sec to blur %s\n", t.GetSeconds(), it->first.c_str());
}

void ModelBank::Blur() {
  std::vector<std::thread> threads;
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    threads.emplace_back(&ModelBank::BlurClass, this, it->first);
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void ModelBank::SaveModelBank(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}


ModelBank ModelBank::LoadModelBank(const char *fn) {
  ModelBank mb;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> mb;

  return mb;
}

} // namespace kitti
} // namespace app
