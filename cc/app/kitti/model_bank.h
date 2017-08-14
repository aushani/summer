#pragma once

#include <map>
#include <string.h>

#include "library/kitti/velodyne_scan.h"

#include "app/kitti/ray_model.h"
#include "app/kitti/observation.h"

#include <boost/serialization/map.hpp>

namespace kt = library::kitti;

namespace app {
namespace kitti {

class ModelBank {
 public:
  ModelBank();

  void MarkObservations(const ObjectState &os, const std::vector<Observation> &x_hits);

  const RayModel& GetModel(const std::string &name) const;
  const std::map<std::string, RayModel>& GetModels() const;

  std::vector<ModelObservation> GetRelevantModelObservations(const std::vector<ModelObservation> &mos) const;

  double EvaluateScan(const ObjectState &os, const kt::VelodyneScan &scan) const;
  double EvaluateModelObservations(const ObjectState &os, const std::vector<ModelObservation> &mos) const;

  void PrintStats() const;
  std::vector<std::string> GetClasses() const;

  void Blur();

  void SaveModelBank(const char *fn) const;
  static ModelBank LoadModelBank(const char *fn);

  double GetMaxSizeXY() const;
  double GetMaxSizeZ() const;

 private:
  // These are the object models across different classes
  std::map<std::string, RayModel> obj_models_;

  void BlurClass(const std::string &classname);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & obj_models_;
  }

};

} // namespace kitti
} // namespace app
