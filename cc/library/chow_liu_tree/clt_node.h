#pragma once

#include <vector>
#include <memory>

#include <boost/optional.hpp>

#include "library/chow_liu_tree/conditional_distribution.h"
#include "library/chow_liu_tree/marginal_distribution.h"
#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

class CLTNode : public std::enable_shared_from_this<CLTNode> {
 public:
  CLTNode(const rt::Location &loc, const JointModel &jm);

  bool HasParent() const;
  void SetParent(const std::shared_ptr<CLTNode> &parent, const JointModel &jm);
  const std::shared_ptr<CLTNode>& GetParent() const;

  const std::vector<std::shared_ptr<CLTNode> >& GetChildren() const;

  const rt::Location& GetLocation() const;

  double GetMarginalLogProb(bool occu) const;
  double GetConditionalLogProb(bool occu, bool parent) const;

 private:
  rt::Location loc_;
  std::vector<std::shared_ptr<CLTNode> > children_;

  std::shared_ptr<CLTNode> parent_;

  MarginalDistribution marginal_;
  boost::optional<ConditionalDistribution> conditional_;
};

} // namespace chow_liu_tree
} // namespace library
