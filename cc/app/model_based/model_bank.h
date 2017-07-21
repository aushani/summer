#pragma once

#include <vector>
#include <map>
#include <string.h>

#include <Eigen/Core>

#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "ray_model.h"
#include "histogram.h"

class ModelBank {
 public:
  ModelBank();

  void AddRayModel(const std::string &name, double size, double p_obj);

  void MarkObservation(const std::string &name, const Eigen::Vector2d &x_sensor_object, double object_angle, const Eigen::Vector2d &x_hit);

  const std::map<std::string, RayModel>& GetModels() const;
  const RayModel& GetModel(const std::string &name) const;

  std::map<std::string, double> EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits) const;
  double EvaluateObservations(const Eigen::Vector2d &x_sensor_object, double object_angle, const std::vector<Eigen::Vector2d> &x_hits, const std::string &classname) const;

  double GetProbObj(const std::string &name) const;

  std::vector<std::string> GetClasses() const;

 private:
  // These are the object models
  std::map<std::string, RayModel> obj_models_;

  std::map<std::string, double> p_objs_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & obj_models_;
    ar & p_objs_;
  }
};
