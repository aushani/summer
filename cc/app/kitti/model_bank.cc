#include "app/kitti/model_bank.h"

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
  it->second.MarkObservations(os, mos);
}

const RayModel& ModelBank::GetModel(const std::string &name) const {
  // Assume class exists
  return obj_models_.find(name)->second;
}

void ModelBank::PrintStats() const {
  for (auto it = obj_models_.begin(); it != obj_models_.end(); it++) {
    printf("Class %s\n", it->first.c_str());
    it->second.PrintStats();
  }
}

} // namespace kitti
} // namespace app
