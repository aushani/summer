#pragma once

#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

struct ConditionalDistribution {
  float log_p[4] = {0.0, 0.0, 0.0, 0.0};
  float mutual_information = 0.0;

  ConditionalDistribution(const rt::Location &loc, const rt::Location &loc_parent, const JointModel &jm) {
    for (int i=0; i<2; i++) {
      bool occ = i==0;

      for (int j=0; j<2; j++) {
        bool parent = j==0;

        int count = jm.GetCount(loc, occ, loc_parent, parent);
        int count_other = jm.GetCount(loc, !occ, loc_parent, parent);
        double denom = count + count_other;

        log_p[GetIndex(occ, parent)] = log(count/denom);
      }
    }

    mutual_information = jm.GetMutualInformation(loc, loc_parent);
  }

  double GetMutualInformation() const {
    return mutual_information;
  }

  double GetLogProb(bool occ, bool given) const {
    return log_p[GetIndex(occ, given)];
  }

  size_t GetIndex(bool occ, bool given) const {
    size_t idx = 0;
    if (occ) {
      idx += 1;
    }

    if (given) {
      idx += 2;
    }

    return idx;
  }
};

} // namespace chow_liu_tree
} // namespace library
