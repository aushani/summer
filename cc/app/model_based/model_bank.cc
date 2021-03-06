#include "app/model_based/model_bank.h"

namespace app {
namespace model_based {

ModelBank::ModelBank() {
}

ModelBank::ModelBank(const ModelBank &mb) :
  obj_models_(mb.obj_models_),
  p_objs_(mb.p_objs_) {
}

void ModelBank::AddRayModel(const std::string &name, double size, double p_obj) {
  obj_models_.emplace(name, size);
  p_objs_[name] = p_obj;
}

void ModelBank::AddRayModel(const std::string &name, double size, double phi_step, double distance_step, double p_obj) {
  obj_models_.insert( std::pair<std::string, RayModel>(name, RayModel(size, phi_step, distance_step)) );
  p_objs_[name] = p_obj;
}

void ModelBank::MarkObservations(const ObjectState &os, const std::vector<Observation> &x_hits) {
  auto it = obj_models_.find(os.GetClassname());
  if (it == obj_models_.end()) {
    printf("No model found for %s\n", os.GetClassname().c_str());
  } else {
    it->second.MarkObservationsWorldFrame(os, x_hits);
  }
}

void ModelBank::UseNGram(int n_gram) {
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    it->second.UseNGram(n_gram);
  }
}

double ModelBank::GetProbObj(const std::string &name) const {
  auto it = p_objs_.find(name);
  if (it == p_objs_.end()) {
    return 0.0;
  }
  return it->second;
}

const std::map<std::string, RayModel>& ModelBank::GetModels() const {
  return obj_models_;
}

const RayModel& ModelBank::GetModel(const std::string &name) const {
  // Assume class exists
  return obj_models_.find(name)->second;
}

double ModelBank::EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const {
  auto it = obj_models_.find(os.GetClassname());
  if (it != obj_models_.end()) {
    return it->second.EvaluateObservations(os, x_hits);
  }

  return 0.0f;
}

std::vector<std::string> ModelBank::GetClasses() const {
  std::vector<std::string> classes;
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    classes.push_back(it->first);
  }

  return classes;
}

void ModelBank::PrintStats() const {
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    printf("Class: %s\n", it->first.c_str());
    it->second.PrintStats();
    printf("\n");
  }
}

} // namespace model_based
} // namespace app
