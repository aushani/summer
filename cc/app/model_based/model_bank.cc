#include "model_bank.h"

ModelBank::ModelBank() {

}

void ModelBank::AddRayModel(const std::string &name, double size) {
  models_.emplace(name, size);
}

void ModelBank::MarkObservation(const std::string &name, const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit) {
  auto it = models_.find(name);
  if (it == models_.end()) {
    printf("No model found for %s\n", name.c_str());
  } else {
    it->second.MarkObservationWorldFrame(x_sensor_object, object_angle, x_hit);
  }
}

const std::map<std::string, RayModel>& ModelBank::GetModels() const {
  return models_;
}

std::map<std::string, double> ModelBank::EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const {
  std::map<std::string, double> result;

  for (auto it = models_.begin(); it != models_.end(); it++) {
    result[it->first] = it->second.EvaluateObservations(x_sensor_object, object_angle, x_hits);
  }

  return result;
}
