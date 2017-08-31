#pragma once

#include "library/ray_tracing/occ_grid_location.h"

#include "library/chow_lui_tree/joint_model.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class Node {
 public:
  Node(const rt::Location &my_loc, const JointModel &jm);
  Node(const rt::Location &my_loc, const Node &parent, const JointModel &jm);

 private:
  struct MarginalDistribution {

  };

  struct ConditionalDistribution {

  };

  rt::Location location_;

  boost::optional<std::shared_ptr<Node>> parent_;

};

}
}
