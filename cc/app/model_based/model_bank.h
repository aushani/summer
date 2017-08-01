#pragma once

#include <vector>
#include <map>
#include <string.h>

#include <Eigen/Core>

#include <boost/serialization/map.hpp>

#include "ray_model.h"
#include "histogram.h"
#include "observation.h"
#include "object_state.h"

class ModelBank {
 public:
  ModelBank();

  void AddRayModel(const std::string &name, double size, double p_obj);
  void AddRayModel(const std::string &name, double size, double phi_step, double distance_step, double p_obj);

  void MarkObservations(const ObjectState &os, const std::vector<Observation> &x_hits);

  const std::map<std::string, RayModel>& GetModels() const;
  const RayModel& GetModel(const std::string &name) const;

  double EvaluateObservations(const ObjectState &os, const std::vector<Observation> &x_hits) const;

  double GetProbObj(const std::string &name) const;

  std::vector<std::string> GetClasses() const;

  void PrintStats() const;

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
