#pragma once

#include <vector>
#include <map>
#include <string.h>

#include <Eigen/Core>

#include "ray_model.h"

class ModelBank {
 public:
  ModelBank();

  void AddRayModel(const std::string &name, double size);

  void MarkObservation(const std::string &name, const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

  const std::map<std::string, RayModel>& GetModels() const;

  std::map<std::string, double> EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const;

 private:
  std::map<std::string, RayModel> models_;


};
