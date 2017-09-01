#pragma once

#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

struct MarginalDistribution {
  float log_p[2] = {0.0, 0.0};

  MarginalDistribution(const rt::Location &loc, const JointModel &jm) {
    int c_t = jm.GetCount(loc, true);
    int c_f = jm.GetCount(loc, false);
    double denom = jm.GetNumObservations(loc);

    log_p[GetIndex(true)] = log(c_t/denom);
    log_p[GetIndex(false)] = log(c_f/denom);
  }

  double GetLogProb(bool occ) const {
    return log_p[GetIndex(occ)];
  }

  size_t GetIndex(bool occ) const {
    return occ ? 0:1;
  }
};

} // namespace chow_liu_tree
} // namespace library
