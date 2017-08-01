#include "model_bank.h"

ModelBank::ModelBank() {
}

void ModelBank::AddRayModel(const std::string &name, double size, double p_obj) {
  obj_models_.emplace(name, size);
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

void ModelBank::MarkEmptyObservation(const ObjectState &os, const Observation &x_hit) {
  empty_model_.MarkObservationWorldFrame(os, x_hit);
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
  return obj_models_.find(name)->second;
}

const EmptyModel& ModelBank::GetEmptyModel() const {
  return empty_model_;
}

double ModelBank::EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const {
  auto it = obj_models_.find(os.GetClassname());
  if (it != obj_models_.end()) {
    return it->second.EvaluateObservations(os, x_hits);
  }

  return empty_model_.EvaluateObservations(os, x_hits);
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

  printf("EMPTY\n");
  empty_model_.PrintStats();
  printf("\n");
}
