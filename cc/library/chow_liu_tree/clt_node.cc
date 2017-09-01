#include "library/chow_liu_tree/clt_node.h"

#include <boost/assert.hpp>

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

CLTNode::CLTNode(const rt::Location &loc) :
 loc_(loc) {}

CLTNode::CLTNode(const rt::Location &loc, CLTNode *parent) :
 loc_(loc), parent_(parent) {
  parent_->children_.push_back(shared_from_this());
}

bool CLTNode::HasParent() const {
  if (parent_) {
    return true;
  }
  return false;
}

void CLTNode::SetParent(const std::shared_ptr<CLTNode> &parent) {
  parent_ = parent;
  parent_->children_.push_back(shared_from_this());
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

} // namespace chow_liu_tree
} // namespace library
