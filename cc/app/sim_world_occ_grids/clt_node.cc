#include "app/sim_world_occ_grids/clt_node.h"

namespace rt = library::ray_tracing;

namespace app {
namespace sim_world_occ_grids {

CLTNode::CLTNode(const rt::Location &loc, const JointModel &jm) :
 loc_(loc), marginal_(loc, jm) {
}

CLTNode::CLTNode(const rt::Location &loc, const CLTNode &parent_node, const JointModel &jm) :
 CLTNode(loc, jm) {
  // Add immediate parent to my list of ancestors
  ancestors_locs_.push_back(parent_node.GetLocation());
  ancestors_dists_.emplace_back(loc_, parent_node.GetLocation(), jm);

  // Go through my parent's ancestors
  for (const auto &ancestor_loc : parent_node.ancestors_locs_) {
    ancestors_locs_.push_back(ancestor_loc);
    ancestors_dists_.emplace_back(loc_, ancestor_loc, jm);
  }
}

void CLTNode::AddChild(const rt::Location &child_loc) {
  children_locs_.push_back(child_loc);
}

size_t CLTNode::NumAncestors() const {
  return ancestors_locs_.size();
}

const rt::Location& CLTNode::GetAncestorLocation(size_t levels) const {
  BOOST_ASSERT(levels < NumAncestors());
  return ancestors_locs_[levels];
}

const rt::Location& CLTNode::GetLocation() const {
  return loc_;
}

const std::vector<rt::Location>& CLTNode::GetChildrenLocations() const {
  return children_locs_;
}

//double CLTNode::GetMutualInformation(size_t ancestor_level) const {
//  double mi = 0;
//
//  for (int i=0; i<2; i++) {
//    bool ancestor_occu = (i == 0);
//    for (int j=0; j<2; j++) {
//      bool node_occu = (j == 0);
//
//      double p_x = GetMarginalProbability(node_occu);
//      double p_y = GetAncestor(ancestor_level).GetMarginalProbability(node_occu);
//      double p_xy = GetConditionalProbability(node_occu, ancestor_level, ancestor_occu);
//
//      mi += p_xy * log(p_xy / (p_x * p_y));
//    }
//  }
//
//  return mi;
//}

double CLTNode::GetConditionalProbability(bool node_occu, size_t ancestor_level, bool ancestor_occu) const {
  BOOST_ASSERT(ancestor_level < NumAncestors());
  const ConditionalDistribution &cd = ancestors_dists_[ancestor_level];

  return cd.GetProbability(node_occu, ancestor_occu);
}

double CLTNode::GetMarginalProbability(bool occu) const {
  return marginal_.GetProbability(occu);
}

}  // namespace sim_world_occ_grids
}  // namespace app
