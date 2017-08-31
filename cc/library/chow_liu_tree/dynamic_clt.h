#pragma once

#include "library/chow_liu_tree/joint_model.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class DynamicCLT {
 public:
  DynamicCLT(const JointModel &jm);

  double BuildAndEvaluate() const;
  double EvaluateMarginal() const;

 private:
  struct MarginalDistribution {
    float log_p[2] = 0;

    double GetLogProb(bool occ) {
      return log_p[GetIndex(occ)];
    }

    size_t GetIndex(bool occ) const {
      return occ ? 0:1;
    }
  };

  struct ConditionalDistribution {
    MarginalDistribution[2] log_p_cond;

    double GetLogProb(bool occ, bool given) {
      return log_p_cond[GetIndex(given)].GetLogProb(occ);
    }

    size_t GetIndex(bool given) const {
      return given ? 0:1;
    }
  };

  std::vector<Edge> all_edges_;

}:

} // namespace chow_liu_tree
} // namespace library
