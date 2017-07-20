#include "model_bank.h"

ModelBank::ModelBank() :
 obs_model_(kMaxRange_) {

}

void ModelBank::AddRayModel(const std::string &name, double size, double p_obj) {
  obj_models_.emplace(name, size);
  p_objs_[name] = p_obj;
}

void ModelBank::MarkObservation(const std::string &name, const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  auto it = obj_models_.find(name);
  if (it == obj_models_.end()) {
    printf("No model found for %s\n", name.c_str());
  } else {
    it->second.MarkObservationWorldFrame(x_sensor_object, object_angle, x_hit);
    //obs_model_.MarkObservationWorldFrame(x_sensor_object, object_angle, x_hit);
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
  return obj_models_.find(name)->second;
}

std::map<std::string, double> ModelBank::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const {
  std::map<std::string, double> result;

  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    result[it->first] = it->second.EvaluateObservations(x_sensor_object, object_angle, x_hits, obs_model_);
  }

  return result;
}

const RayModel& ModelBank::GetObservationModel() const {
  return obs_model_;
}

void ModelBank::BuildObservationModel() {
  std::vector<RayModel*> models;
  std::vector<double> probs;
  for (auto &it : obj_models_) {
    models.push_back(&it.second);
    probs.push_back(p_objs_[it.first]);
  }

  obs_model_ = RayModel(models, probs);
}
