#pragma once

#include <map>

#include <boost/serialization/map.hpp>

#include "library/ray_tracing/occ_grid_location.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

class MarginalModel {
 public:
  MarginalModel(const JointModel &jm);

  double GetLogProbability(const rt::Location &loc, bool occ) const;

  void Save(const char *fn) const;
  static MarginalModel Load(const char *fn);

 private:
  MarginalModel() {} // for boost serialization

  struct MarginalDistribution {
    double p_occu = 0.0;
    double p_free = 0.0;

    double log_p_occu = 0.0;
    double log_p_free = 0.0;

    int total_count = 0;

    MarginalDistribution() {}; // for boost serialization

    MarginalDistribution(const JointModel &jm, const rt::Location &loc) {
      int c_t = jm.GetCount(loc, true);
      int c_f = jm.GetCount(loc, false);

      total_count = c_t + c_f;

      double denom = c_t + c_f;
      p_occu = c_t / denom;
      p_free = c_f / denom;

      log_p_occu = log(p_occu);
      log_p_free = log(p_free);
    }

    double GetProbability(bool occ) const {
      return occ ? p_occu:p_free;
    }

    double GetLogProbability(bool occ) const {
      return occ ? log_p_occu:log_p_free;
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */){
      ar & p_occu;
      ar & p_free;

      ar & log_p_occu;
      ar & log_p_free;

      ar & total_count;
    }
  };

  double resolution_;

  std::map<rt::Location, MarginalDistribution> model_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & resolution_;
    ar & model_;
  }

};

} // namespace kitti_occ_grids
} // namespace app
