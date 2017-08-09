#pragma once

#include <map>
#include <string.h>

#include "app/kitti/ray_model.h"
#include "app/kitti/observation.h"

#include <boost/serialization/map.hpp>

namespace app {
namespace kitti {

class ModelBank {
 public:
  ModelBank();

  void MarkObservations(const ObjectState &os, const std::vector<Observation> &x_hits);

  const RayModel& GetModel(const std::string &name) const;

  void PrintStats() const;

 private:
  // These are the object models across different classes
  std::map<std::string, RayModel> obj_models_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & obj_models_;
  }

};

} // namespace kitti
} // namespace app
