#include "library/chow_liu_tree/clt_node.h"

#include <boost/assert.hpp>

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

CLTNode::CLTNode(const rt::Location &loc, const JointModel &jm) :
 loc_(loc), marginal_(loc, jm) {
}

bool CLTNode::HasParent() const {
  if (parent_) {
    return true;
  }
  return false;
}

void CLTNode::SetParent(const std::shared_ptr<CLTNode> &parent, const JointModel &jm) {
  // Make sure we haven't previously assigned a parent
  BOOST_ASSERT(!HasParent());

  parent_ = parent;
  parent_->children_.push_back(shared_from_this());
  conditional_ = ConditionalDistribution(loc_, parent->GetLocation(), jm);

  mutual_information_ = jm.GetMutualInformation(loc_, parent_->GetLocation());
}

const std::shared_ptr<CLTNode>& CLTNode::GetParent() const {
  BOOST_ASSERT(HasParent());
  return parent_;
}

const std::vector<std::shared_ptr<CLTNode> >& CLTNode::GetChildren() const {
  return children_;
}

const rt::Location& CLTNode::GetLocation() const {
  return loc_;
}

double CLTNode::GetMarginalLogProb(bool occu) const {
  return marginal_.GetLogProb(occu);
}

double CLTNode::GetConditionalLogProb(bool occu, bool parent) const {
  BOOST_ASSERT(HasParent());
  return conditional_->GetLogProb(occu, parent);
}

double CLTNode::GetNodeMutualInformation() const {
  return mutual_information_;
}

double CLTNode::GetTreeMutualInformation() const {
  double mi = mutual_information_;

  for (const auto &c : children_) {
    mi += c->GetTreeMutualInformation();
  }

  return mi;
}

} // namespace chow_liu_tree
} // namespace library
